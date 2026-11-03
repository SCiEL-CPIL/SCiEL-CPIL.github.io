"""Rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
import time
import random
import itertools

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill

    # The multiplication tensor T[i,j,k] = 1 iff (i,j) in U, (j,k) in V, (k,i) in W
    # Non-zero entries: (i, j, k) for i in range(n), j in range(m), k in range(p)
    # Tensor shape: (n*m) x (m*p) x (p*n) with index convention:
    #   U index: i*m + j
    #   V index: j*p + k
    #   W index: k*n + i
    nz = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]
    num_nz = len(nz)  # 84

    def make_schoolbook():
        rank = num_nz
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c, (ui, vi, wi) in enumerate(nz):
            U[ui][c] = 1
            V[vi][c] = 1
            W[wi][c] = 1
        return {"U": U, "V": V, "W": W}

    best = make_schoolbook()
    best_rank = num_nz

    def verify(U, V, W, rank):
        """Check exact decomposition: sum_c U[a,c]*V[b,c]*W[c_index] == T[a,b,c_index]"""
        # For each (a,b,c) triple, check if sum over columns equals the tensor value
        # Tensor value is 1 if (a,b,c) is in nz, else 0
        nz_set = set(nz)
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    expected = 1 if (a, b, c) in nz_set else 0
                    actual = 0
                    for col in range(rank):
                        actual += U[a][col] * V[b][col] * W[c][col]
                    if actual != expected:
                        return False
        return True

    def verify_fast(U, V, W, rank):
        """Faster verification using only non-zero positions and checking zeros."""
        nz_set = set(nz)
        all_triples = set()
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    all_triples.add((a, b, c))
        
        # Check all non-zero positions
        for (a, b, c) in nz:
            actual = 0
            for col in range(rank):
                if U[a][col] != 0 and V[b][col] != 0 and W[c][col] != 0:
                    actual += U[a][col] * V[b][col] * W[c][col]
            if actual != 1:
                return False
        
        # Check a sample of zero positions (all if fast enough)
        zero_triples = all_triples - nz_set
        # Check all zero triples for correctness
        for (a, b, c) in zero_triples:
            actual = 0
            for col in range(rank):
                if U[a][col] != 0 and V[b][col] != 0 and W[c][col] != 0:
                    actual += U[a][col] * V[b][col] * W[c][col]
            if actual != 0:
                return False
        return True

    # Greedy set cover: each rank-1 tensor is determined by (support_U, support_V, support_W)
    # A rank-1 tensor with U support Su, V support Sv, W support Sw covers triple (a,b,c)
    # iff a in Su, b in Sv, c in Sw, and the product is 1.
    # For exact decomposition with ±1 entries, we need the sum to work out.
    
    # Strategy: build columns greedily. Each column picks a subset of nz to "explain".
    # A column (u, v, w) where u is a 0/1 vector on n*m indices, v on m*p, w on p*n
    # contributes u[a]*v[b]*w[c] to position (a,b,c).
    # For the sum to equal the tensor, we need careful construction.
    
    # Simpler approach: try to find rank-r decomposition where r < 84
    # Use the fact that the tensor has structure.
    
    # Let's try: for each column, pick a "core" triple and extend.
    # Actually, let's use a different approach:
    # The tensor T = sum_c u_c v_c w_c where u_c, v_c, w_c are vectors.
    # For binary (0/1) entries, this is like a set system.
    
    # Key observation: T[i*m+j, j*p+k, k*n+i] = 1
    # This means the tensor is determined by the "chain" i->j->k->i.
    
    # Let's try to find a low-rank decomposition using the structure.
    # For 3x4x7, the schoolbook is 84. Can we do better?
    
    # Try random search: pick random rank-r, random ±1 entries, verify.
    # This is unlikely to work for r much less than 84.
    
    # Better: use the structure. Note that for fixed j, the sub-tensor over (i,k)
    # is rank 1 (it's the outer product of identity-like structures).
    # Actually T[i*m+j, j*p+k, k*n+i] = 1 for all i,k when j is fixed.
    # So for fixed j, the (i,k) slice is all-ones matrix of size n x p.
    
    # The tensor can be written as sum over j of (e_j in middle) tensor (all-ones in i,k with the W index coupling).
    # But the W index couples k and i: W index is k*n+i.
    
    # Let me think differently. The tensor is:
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # i.e., a mod m = b // p = j, and b mod p = k, and c mod n = i, and a // m = i, c // n = k
    # So T[a,b,c] = 1 iff (a//m == c%n) and (a%m == b//p) and (b%p == c//n)
    
    # This is a structured tensor. Let's try to find a decomposition.
    
    # Approach: use the "chain" structure. For each (i,j,k), we have one non-zero.
    # The minimal rank is at least the maximum of:
    #   - rank of any (a,b) slice over c: for fixed a=i*m+j, b=j*p+k, only c=k*n+i is non-zero -> rank 1
    #   - rank of any (a,c) slice over b: for fixed a=i*m+j, c=k*n+i, only b=j*p+k is non-zero -> rank 1
    #   - rank of any (b,c) slice over a: for fixed b=j*p+k, c=k*n+i, only a=i*m+j is non-zero -> rank 1
    # So the tensor has "slice rank" 1 in all directions, meaning it could potentially be low rank.
    
    # Actually, the tensor is a sum of 84 rank-1 tensors (the schoolbook). Can we combine them?
    # Two rank-1 tensors can combine if they share structure.
    
    # Let's try: group by j. For fixed j, the non-zones are (i*m+j, j*p+k, k*n+i) for all i,k.
    # This is a "permutation-like" structure: it's the tensor with T[i,k] = 1 for all i,k,
    # but with the W index being k*n+i (not k*n+something independent).
    
    # For fixed j, define U_j[i] = 1 (for the i component), V_j[k] = 1, W_j[k,i] = 1.
    # But W_j is not a rank-1 tensor in (k,i) space - it's all ones, which IS rank 1!
    # Wait, W index is k*n+i, so W_j as a vector over (k*n+i) has 1 at positions k*n+i for all k,i.
    # That's n*p = 21 ones. As a vector, it's not rank-1 in any meaningful sense for tensor decomposition.
    
    # Let me reconsider. In the tensor decomposition T = sum_c u_c ⊗ v_c ⊗ w_c:
    # u_c is a vector of length n*m, v_c of length m*p, w_c of length p*n.
    # The contribution to (a,b,c_idx) is u_c[a] * v_c[b] * w_c[c_idx].
    
    # For the schoolbook, each column has exactly one 1 in each of u, v, w.
    
    # To reduce rank, we need columns where u, v, w have multiple 1s, such that
    # the product still gives the right pattern.
    
    # Key constraint: if u[a] = 1 and v[b] = 1 and w[c_idx] = 1, then the contribution
    # to (a,b,c_idx) is 1. For this to be correct, (a,b,c_idx) must be in nz, OR
    # other columns must cancel it (but with 0/1 entries, cancellation isn't possible;
    # we'd need negative entries).
    
    # With ±1 entries, we CAN have cancellation. So we need:
    # For each (a,b,c_idx) in nz: sum_c u_c[a]*v_c[b]*w_c[c_idx] = 1
    # For each (a,b,c_idx) not in nz: sum_c u_c[a]*v_c[b]*w_c[c_idx] = 0
    
    # This is a system of equations. Let's try to solve it with random initialization
    # and local search.
    
    # Given time constraints, let's try a few random restarts with greedy construction.
    
    def random_attempt(rank_target, seed):
        rng = random.Random(seed)
        # Initialize with random ±1 entries
        U = [[rng.choice([-1, 1]) for _ in range(rank_target)] for _ in range(n * m)]
        V = [[rng.choice([-1, 1]) for _ in range(rank_target)] for _ in range(m * p)]
        W = [[rng.choice([-1, 1]) for _ in range(rank_target)] for _ in range(p * n)]
        
        # Local search: flip random entries to reduce error
        def compute_error():
            err = 0
            for (a, b, c) in nz:
                s = 0
                for col in range(rank_target):
                    s += U[a][col] * V[b][col] * W[c][col]
                err += (s - 1) ** 2
            # Sample some zero positions
            zero_samples = rng.sample(list(set(itertools.product(range(n*m), range(m*p), range(p*n))) - set(nz)), min(200, (n*m)*(m*p)*(p*n) - num_nz))
            for (a, b, c) in zero_samples:
                s = 0
                for col in range(rank_target):
                    s += U[a][col] * V[b][col] * W[c][col]
                err += s ** 2
            return err
        
        best_err = compute_error()
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            # Flip a random entry
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                a, col = rng.randrange(n * m), rng.randrange(rank_target)
                old = U[a][col]
                U[a][col] = -old
            elif mat == 'V':
                b, col = rng.randrange(m * p), rng.randrange(rank_target)
                old = V[b][col]
                V[b][col] = -old
            else:
                c, col = rng.randrange(p * n), rng.randrange(rank_target)
                old = W[c][col]
                W[c][col] = -old
            
            new_err = compute_error()
            if new_err > best_err:
                # Undo
                if mat == 'U':
                    U[a][col] = old
                elif mat == 'V':
                    V[b][col] = old
                else:
                    W[c][col] = old
            else:
                best_err = new_err
        
        # Check if we found exact solution
        if best_err == 0:
            # Verify exactly
            if verify_fast(U, V, W, rank_target):
                return U, V, W
        return None
    
    # Try ranks from 83 down
    for rank in range(num_nz - 1, max(num_nz - 20, 50), -1):
        if time.time() > deadline - 10:
            break
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            result = random_attempt(rank, seed=rank * 1000 + attempt)
            if result is not None:
                U, V, W = result
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        if best_rank < num_nz - 1:
            break
    
    return best
# EVOLVE-BLOCK-END

