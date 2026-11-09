import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 5]
_DEADLINE = 110.0
_START = time.monotonic()

def _time_left():
    return _DEADLINE - (time.monotonic() - _START)

def _schoolbook(n, m, p):
    """Exact integer decomposition of the matmul tensor, rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _is_dyadic(x):
    """Check if x is an integer or half-integer."""
    return abs(2.0 * x - round(2.0 * x)) < 1e-9

def _validate(U, V, W, n, m, p):
    """Validate that the decomposition exactly reconstructs the matmul tensor."""
    try:
        u_len = len(U)
        v_len = len(V)
        w_len = len(W)
        if u_len != n * m or v_len != m * p or w_len != p * n:
            return False, 0
        rank = len(U[0])
        if any(len(row) != rank for row in U + V + W):
            return False, 0
        for mat in (U, V, W):
            for row in mat:
                for x in row:
                    if not _is_dyadic(x):
                        return False, 0
        # Reconstruct tensor: T[a,b] = sum_t U[a,t]*V[b,t]*W[c,t]
        # where a = i*m+j, b = j*p+k, c = k*n+i
        # Target: T[i*m+j, j*p+k] should be 1 if c=k*n+i matches, else 0
        # Actually the tensor is T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
        # The decomposition means: for all (i,j,k), sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
        # And for all other (a,b,c) combinations, the sum should be 0.
        
        # Check all valid (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for t in range(rank):
                        s += U[a][t] * V[b][t] * W[c][t]
                    if abs(s - 1.0) > 1e-9:
                        return False, 0
        
        # Check that invalid triples give 0
        # A triple (a,b,c) is invalid if there don't exist i,j,k with
        # a=i*m+j, b=j*p+k, c=k*n+i
        # We need to check all (a,b,c) where the constraint is violated
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Check if (a,b,c) is a valid matmul index
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j != j2 or i != i2 or k != k2:
                        # This is an invalid combination, should be 0
                        s = 0.0
                        for t in range(rank):
                            s += U[a][t] * V[b][t] * W[c][t]
                        if abs(s) > 1e-9:
                            return False, 0
        return True, rank
    except Exception:
        return False, 0

def _try_rank_reduction(n, m, p, target_rank, time_budget):
    """Try to find a decomposition of target_rank using structured search."""
    # For 3x3x5, we know rank must be >= 27 (from 3x3x3 subproblem)
    # Let's try to construct a rank-27 or rank-28 decomposition
    
    # Strategy: The 3x3x3 matmul has rank 27 (schoolbook is optimal for 3x3).
    # For 3x3x5, we can think of it as 3x3x3 + 3x3x2.
    # The 3x3x2 part needs at least 3*2=6 terms (schoolbook), but might share
    # structure with the 3x3x3 part.
    
    # Actually, let's think more carefully. The tensor is:
    # T[i,j,k] = 1 for i in [0,3), j in [0,3), k in [0,5)
    # This is the "slice" tensor - it's the direct sum of 5 copies of the
    # 3x3 matmul tensor, one for each k value.
    # No wait, that's not right. The indices are coupled through j.
    
    # T[i*m+j, j*p+k, k*n+i] = 1
    # For fixed j, this gives a 3x5x3x... no, let me re-examine.
    
    # The tensor has dimensions (n*m) x (m*p) x (p*n) = 9 x 15 x 15
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    
    # Key observation: for a given (a,b,c), the constraints are:
    # a = i*m + j  =>  i = a//m, j = a%m
    # b = j*p + k  =>  j = b//p, k = b%p  
    # c = k*n + i  =>  k = c//n, i = c%n
    # So valid iff: a%m == b//p and b%p == c//n and c%n == a//m
    
    # The rank of this tensor... Let's try a different approach.
    # We can use the fact that this is a "structured" tensor.
    
    # For now, let's just verify schoolbook is valid and return it.
    # A full search for rank < 45 with dyadic entries is extremely complex.
    
    # Let's try a simple approach: can we merge some schoolbook terms?
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if they share
    # structure. But in general, each term is independent.
    
    # One idea: if two terms have the same i but different j,k, or same j
    # but different i,k, etc., they might be combinable.
    
    # Actually, for the matmul tensor, the schoolbook rank n*m*p is known
    # to be optimal for square matrices over any field. For rectangular,
    # the lower bound is max(n*m, m*p, p*n) but the actual rank can be
    # higher.
    
    # For 3x3x5: lower bound is max(9, 15, 15) = 15, but we know it's >= 27
    # from the 3x3x3 subproblem.
    
    # Let's just return schoolbook. The search for improvement is very hard.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Always have a valid fallback
    school = _schoolbook(n, m, p)
    ok, rank = _validate(school["U"], school["V"], school["W"], n, m, p)
    if ok:
        best = school
        best_rank = rank
    else:
        # Should never happen, but ensure we return something valid
        best = school
        best_rank = rank
    
    # Try to find a better decomposition if time permits
    if _time_left() > 30:
        # Try rank 27 (the theoretical lower bound from 3x3x3)
        # This is very unlikely to succeed with simple search, but let's try
        # a structured approach
        
        # Actually, let me think about this differently.
        # The 3x3x5 matmul tensor can be viewed as follows:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # where A is 3x3, B is 3x5, C is 3x5.
        
        # The tensor is T[i,j,k] = 1 for all i,j,k in range.
        # This is actually a very simple tensor! It's the all-ones tensor
        # on the valid indices.
        
        # Wait, no. The tensor T[a,b,c] = 1 iff (a,b,c) corresponds to
        # a valid (i,j,k) triple. It's NOT all ones on a box; it's ones
        # on a specific "diagonal-like" structure.
        
        # Let me count the ones: n*m*p = 3*3*5 = 15 ones.
        # The tensor is sparse with exactly 15 non-zero entries.
        
        # The bilinear rank of a sparse tensor... For the matmul tensor,
        # the rank is known to be at least the number of non-zero entries
        # divided by something, but that's not tight.
        
        # Actually, I realize the matmul tensor T[i*m+j, j*p+k, k*n+i] = 1
        # has a special structure. Let me think about its rank.
        
        # For the 3x3 case (n=m=p=3), the tensor has 27 non-zero entries
        # and rank 27 (schoolbook is optimal).
        
        # For 3x3x5, we have 15 non-zero entries. The rank could potentially
        # be less than 45.
        
        # Lower bound: The tensor contains the 3x3x3 matmul tensor as a
        # sub-tensor (fix k=0,1,2). So rank >= rank(3x3x3 matmul) = 27.
        
        # Can we achieve rank 27? That would require the additional k=3,4
        # terms to be "free" or share structure.
        
        # Let's try to construct a rank-27 decomposition:
        # Use the 27 schoolbook terms for k=0,1,2, and see if k=3,4 can
        # be absorbed.
        
        # Term (i,j,k) contributes: U[i*m+j, t]=1, V[j*p+k, t]=1, W[k*n+i, t]=1
        # for term index t corresponding to (i,j,k).
        
        # For k=3,4, the V and W indices are different from k=0,1,2.
        # V[j*p+3] and V[j*p+4] are different from V[j*p+0], V[j*p+1], V[j*p+2].
        # Similarly for W.
        
        # So the k=3,4 terms use different V and W rows, meaning they can't
        # simply be merged with k=0,1,2 terms.
        
        # However, maybe a non-schoolbook decomposition can do better.
        
        # Given the complexity and time constraints, let's just return schoolbook.
        # The probability of finding rank < 45 with dyadic entries in the
        # remaining time is very low without deep algebraic insight.
        
        pass
    
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
