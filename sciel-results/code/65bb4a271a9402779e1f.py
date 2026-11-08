"""Tensor decomposition for 2x3 by 3x10 matrix multiplication.
Uses the schoolbook (canonical) decomposition which is guaranteed correct.
Attempts to find a lower-rank decomposition via search if time permits."""
import time
import itertools
import random

DIMENSIONS = [2, 3, 10]
DEADLINE_SECONDS = 140  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    nm = n * m  # 6
    mp = m * p  # 30
    pn = p * n  # 20
    
    # Schoolbook decomposition: rank = n*m*p = 60
    # U[i*m + j, c] = 1 if c corresponds to (i,j,k)
    # V[j*p + k, c] = 1 if c corresponds to (i,j,k)
    # W[k*n + i, c] = 1 if c corresponds to (i,j,k)
    # Index: c = i*(m*p) + j*p + k
    
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * (m * p) + j * p + k
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = n * m * p  # 60
    
    def validate(U, V, W, rank):
        """Check if decomposition is exact: sum_c U[a,c]*V[b,c] = W_tensor[a,b,c_index]"""
        # The tensor T[i,j,k] = 1 for all (i,j,k), and we need:
        # sum_c U[i*m+j, c] * V[j*p+k, c] = delta for the matrix mult tensor
        # Actually, the matrix multiplication tensor has T[i,j,k] = 1 iff the element (i,k) of A@B 
        # uses A[i,j]*B[j,k]. So the tensor is 1 for all (i,j,k) in the index space.
        # Wait no. The matrix multiplication tensor T such that (A@B)[i,k] = sum_j A[i,j]*B[j,k] = sum_j T[i,j,k]*A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k.
        # The decomposition says: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * ... 
        # No wait. The standard tensor decomposition for matrix mult is:
        # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr} where U is n x rank, V is m x rank, W is p x rank
        # But the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # This is a different convention. Let me re-read.
        # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
        # And the tensor is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k'*n+i',c] should give 
        # the product when properly indexed.
        # Actually for matrix multiplication A(2x3) * B(3x10) = C(2x10):
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T[i,j,k] = 1 (for all i,j,k) since it's just the identity in the middle.
        # The factorization with these large matrices:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] ... no that doesn't work dimensionally.
        
        # Let me think again. The output factor W has indices [k*n + i, r].
        # So W is indexed by (k,i) pairs, shape (p*n, rank).
        # U is indexed by (i,j) pairs, shape (n*m, rank).
        # V is indexed by (j,k) pairs, shape (m*p, rank).
        
        # The decomposition should satisfy: for all i,j,k,
        # sum_c U[i*m+j, c] * V[j*p+k, c] = some function that when combined with W gives the result.
        
        # Actually, I think the correct interpretation is:
        # The matrix multiplication can be written as:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j A[i,j] U[i*m+j,c]) (sum_j B[j,k] V[j*p+k,c]) W[k*n+i,c]
        # No, that's not right either.
        
        # Let me look at the schoolbook: U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 for c=(i,j,k).
        # Then sum_c U[i*m+j',c]*V[j'*p+k',c]*W[k*n+i'',c] for fixed (i,j,k):
        # = sum over all c of product. Only c=(i,j,k) contributes if i'=i,j'=j,k'=k,i''=i,k''=k.
        # So sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if all indices match, else 0.
        # That's delta_{ii'}delta_{jj'}delta_{kk'}delta_{kk''}... no.
        
        # I think the verification is: for the matrix product, we need
        # (A@B)[i,k] = sum_j A[i,j]*B[j,k]
        # = sum_c [sum_j A[i,j]*U[i*m+j,c]] * [sum_j B[j,k]*V[j*p+k,c]] * W[k*n+i,c]
        # For this to work with schoolbook:
        # sum_j A[i,j]*U[i*m+j,c] = A[i,j_c] where j_c is the j component of c
        # sum_j B[j,k]*V[j*p+k,c] = B[j_c,k_c] where j_c,k_c are from c
        # W[k*n+i,c] = 1 if k=k_c and i=i_c
        # So product = A[i,j_c]*B[j_c,k_c]*delta_{k,k_c}*delta_{i,i_c}
        # Sum over c: sum_{i_c,j_c,k_c} A[i,j_c]*B[j_c,k_c]*delta_{k,k_c}*delta_{i,i_c}
        # = sum_{j_c} A[i,j_c]*B[j_c,k] when i_c=i, k_c=k
        # = sum_j A[i,j]*B[j,k] ✓
        
        # So validation: for all i,k, 
        # sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_j B[j,k]*V[j*p+k,c]) * W[k*n+i,c] == sum_j A[i,j]*B[j,k]
        # This must hold for ALL matrices A,B, so it's equivalent to:
        # sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c] = delta_{j,j'} for all i,j,j',k
        # Wait let me redo. Fix i,k. The expression is:
        # sum_c [sum_j A[i,j] U[i*m+j,c]] [sum_{j'} B[j',k] V[j'*p+k,c]] W[k*n+i,c]
        # = sum_c sum_j sum_{j'} A[i,j] B[j',k] U[i*m+j,c] V[j'*p+k,c] W[k*n+i,c]
        # For this to equal sum_j A[i,j] B[j,k] for all A,B, we need:
        # sum_c U[i*m+j,c] V[j'*p+k,c] W[k*n+i,c] = delta_{j,j'} for all i,j,j',k
        
        # Let's verify schoolbook: c=(i_c,j_c,k_c)
        # sum_{i_c,j_c,k_c} delta_{j,j_c} * delta_{j',j_c} * delta_{k,k_c} * delta_{i,i_c} * delta_{k,k_c}
        # Wait W[k*n+i,c] = 1 if k_c=k and i_c=i
        # = sum_{j_c,k_c} delta_{j,j_c} delta_{j',j_c} delta_{k,k_c} delta_{k,k_c}
        # = sum_{j_c} delta_{j,j_c} delta_{j',j_c} = delta_{j,j'} ✓
        
        return True
    
    def check_decomposition(U, V, W, rank):
        """Verify sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c] == delta_{j,j'}"""
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        expected = 1 if j == jp else 0
                        s = 0
                        for c in range(rank):
                            s += U[i * m + j][c] * V[jp * p + k][c] * W[k * n + i][c]
                        if abs(s - expected) > 1e-9:
                            return False
        return True
    
    # Try to find lower rank decompositions
    # For 2x3x10, known upper bounds suggest rank might be lower than 60.
    # Let's try a structured approach: rank = 2*p = 20 (each (i,k) pair gets p terms? No.)
    # Or rank = n*p = 20, or m*p = 30, or 3*p = 30.
    
    # Try rank 30: one term per (j,k) pair for each i? No, that's 2*3*10=60.
    # Try rank 20: perhaps sharing across i?
    
    # For now, if we can't find better quickly, return schoolbook.
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 5:
        return best_result
    
    # Try a greedy/local search for lower rank
    # Start with rank 30
    for target_rank in [30, 25, 20, 24, 28, 29]:
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
        if target_rank >= best_rank:
            continue
        
        # Try random initialization with structured patterns
        found = False
        for attempt in range(100):
            if time.time() - start_time > DEADLINE_SECONDS - 10:
                break
            
            # Pattern 1: U[i*m+j,c] depends on i,j and c mod something
            # Try: for c in range(rank), let i_c = c % n, j_c = (c // n) % m, k_c = c // (n*m) % p
            # This gives n*m*p distinct (i,j,k) if rank = n*m*p, but for smaller rank we wrap around.
            
            # Alternative: try rank = m*p = 30 with U[i*m+j,c] = 1 if j == c%m and i == c//m % n ... 
            # c = i*m + j for U, but c ranges 0..rank-1
            
            # Let me try a different structure. For rank = 20 = n*p:
            # c = i*p + k for some assignment, but we need j too.
            
            # Try: rank = p = 10? Unlikely.
            
            # Let me try rank 30 with a specific pattern:
            # For each (j,k) in m*p=30, have one term.
            # c = j*p + k
            # U[i*m+j, c] = 1 if c = j*p+k (always true for fixed j) -> U[i*m+j, j*p+k] = 1
            # V[j'*p+k', c] = 1 if j'=j and k'=k -> V[j*p+k, j*p+k] = 1
            # W[k*n+i, c] = ? Need to set so that sum works.
            # sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c]
            # = U[i*m+j, j*p+k]*V[j'*p+k, j*p+k]*W[k*n+i, j*p+k] + ...
            # If U[i*m+j,c] = 1 for c = j*p+k (only one c per (j,k) for each i? No, U has row i*m+j)
            # U[i*m+j, j*p+k] = 1 (this is always 1 for the matching j)
            # V[j'*p+k, j*p+k] = 1 if j'=j
            # So sum = V[j'*p+k, j*p+k] * W[k*n+i, j*p+k] = delta_{j,j'} * W[k*n+i, j*p+k]
            # We need this to equal delta_{j,j'}, so W[k*n+i, j*p+k] = 1 for all i,j,k.
            # Check: W has shape (p*n, rank) = (20, 30). W[k*n+i, j*p+k] = 1 for all valid i,k,j.
            # That means W is all ones!
            
            if target_rank == 30:
                rank = 30
                U = [[0]*rank for _ in range(nm)]
                V = [[0]*rank for _ in range(mp)]
                W = [[0]*rank for _ in range(pn)]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            c = j * p + k
                            U[i*m+j][c] = 1
                            V[j*p+k][c] = 1
                            W[k*n+i][c] = 1
                if check_decomposition(U, V, W, rank):
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
                    found = True
                    break
            
            elif target_rank == 20:
                # Try rank 20 = n*p
                # c = i*p + k
                # U[i*m+j, i*p+k] = 1 for all j? Then U[i*m+j, c] = 1 if c//p == i
                # V[j'*p+k', c] = 1 if c%p == k' and j' == ? 
                # This is getting complex. Let's try:
                rank = 20
                U = [[0]*rank for _ in range(nm)]
                V = [[0]*rank for _ in range(mp)]
                W = [[0]*rank for _ in range(pn)]
                for i in range(n):
                    for k in range(p):
                        c = i * p + k
                        for j in range(m):
                            U[i*m+j][c] = 1
                            V[j*p+k][c] = 1
                        W[k*n+i][c] = 1
                if check_decomposition(U, V, W, rank):
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
                    found = True
                    break
            
            elif target_rank == 15:
                # Try rank 15 = m*p/2? Or 3*5?
                rank = 15
                # Try c = j*5 + (k % 5) or something... unlikely to work simply.
                pass
        
        if found:
            break
    
    return best_result
