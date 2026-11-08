import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    def add_strassen_to(U, V, W, i0, j0, k0, c):
        # 7 multiplications for 2x2x2 Strassen
        U[i0*m+j0][c] += 1; U[(i0+1)*m+(j0+1)][c] += 1
        V[j0*p+k0][c] += 1; V[(j0+1)*p+(k0+1)][c] += 1
        W[k0*n+i0][c] += 1; W[(k0+1)*n+(i0+1)][c] += 1
        
        U[(i0+1)*m+j0][c+1] += 1; U[(i0+1)*m+(j0+1)][c+1] += 1
        V[j0*p+k0][c+1] += 1
        W[k0*n+(i0+1)][c+1] += 1; W[(k0+1)*n+(i0+1)][c+1] += -1
        
        U[i0*m+j0][c+2] += 1
        V[j0*p+(k0+1)][c+2] += 1; V[(j0+1)*p+(k0+1)][c+2] += -1
        W[(k0+1)*n+i0][c+2] += 1; W[(k0+1)*n+(i0+1)][c+2] += 1
        
        U[(i0+1)*m+(j0+1)][c+3] += 1
        V[(j0+1)*p+k0][c+3] += 1; V[j0*p+k0][c+3] += -1
        W[k0*n+i0][c+3] += 1; W[k0*n+(i0+1)][c+3] += 1
        
        U[i0*m+j0][c+4] += 1; U[i0*m+(j0+1)][c+4] += 1
        V[(j0+1)*p+(k0+1)][c+4] += 1
        W[k0*n+i0][c+4] += -1; W[(k0+1)*n+i0][c+4] += 1
        
        U[(i0+1)*m+j0][c+5] += 1; U[i0*m+j0][c+5] += -1
        V[j0*p+k0][c+5] += 1; V[j0*p+(k0+1)][c+5] += 1
        W[(k0+1)*n+(i0+1)][c+5] += 1
        
        U[i0*m+(j0+1)][c+6] += 1; U[(i0+1)*m+(j0+1)][c+6] += -1
        V[(j0+1)*p+k0][c+6] += 1; V[(j0+1)*p+(k0+1)][c+6] += 1
        W[k0*n+i0][c+6] += 1

    # Generate all possible placements of 4 disjoint 2x2x2 blocks
    # Block positions: i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3}
    # But blocks must be disjoint
    
    def blocks_disjoint(blocks):
        for a in range(len(blocks)):
            for b in range(a+1, len(blocks)):
                ia, ja, ka = blocks[a]
                ib, jb, kb = blocks[b]
                # Check overlap in i
                if not (ia+1 < ib or ib+1 < ia):
                    # Overlap in i
                    # Check overlap in j
                    if not (ja+1 < jb or jb+1 < ja):
                        # Overlap in j
                        # Check overlap in k
                        if not (ka+1 < kb or kb+1 < ka):
                            # Overlap in all three -> not disjoint
                            return False
        return True

    # All valid block positions
    valid_positions = []
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                valid_positions.append((i0, j0, k0))
    
    # Find all sets of 4 disjoint blocks
    # This is expensive, so let's be smart
    # i0 can be 0 or 1
    # For i0=0: blocks cover i in {0,1}
    # For i0=1: blocks cover i in {1,2}
    # Two blocks with different i0 can coexist only if they don't overlap in i
    # i0=0 and i0=1 overlap in i=1, so they can't coexist unless... wait
    # Block at i0=0 covers i=0,1. Block at i0=1 covers i=1,2. They overlap at i=1.
    # So we can have at most 1 block with i0=0 and at most 1 block with i0=1? No.
    # Actually, blocks with i0=0 all cover i in {0,1}. Any two such blocks overlap in i.
    # So we can have at most 1 block with i0=0 and at most 1 block with i0=1? 
    # No wait, that's not right either. Two blocks both with i0=0 overlap in i dimension.
    # So we can have at most 1 block with i0=0 and at most 1 block with i0=1.
    # That gives at most 2 blocks total? But we have 4 in the current solution!
    
    # Let me re-examine. The current solution has blocks at:
    # (0,0,0), (0,2,0), (0,0,2), (0,2,2)
    # All have i0=0, so they all cover i in {0,1}. They overlap in i!
    # But they don't overlap in j and k.
    # Block (0,0,0): i={0,1}, j={0,1}, k={0,1}
    # Block (0,2,0): i={0,1}, j={2,3}, k={0,1}
    # They overlap in i={0,1} and k={0,1}, but not in j.
    # For the blocks to be "disjoint" in the sense that they cover different tensor entries,
    # we need them to not share any (i,j,k) triple.
    # Two blocks share a triple if they overlap in all three dimensions.
    # So the condition is: for any two blocks, they must not overlap in at least one dimension.
    
    def blocks_disjoint_correct(blocks):
        for a in range(len(blocks)):
            for b in range(a+1, len(blocks)):
                ia, ja, ka = blocks[a]
                ib, jb, kb = blocks[b]
                # They conflict if they overlap in ALL three dimensions
                overlap_i = not (ia+1 < ib or ib+1 < ia)
                overlap_j = not (ja+1 < jb or jb+1 < ja)
                overlap_k = not (ka+1 < kb or kb+1 < ka)
                if overlap_i and overlap_j and overlap_k:
                    return False
        return True

    # Now find all sets of 4 disjoint blocks
    all_sets = []
    for combo in product(range(len(valid_positions)), repeat=4):
        if len(set(combo)) != 4:
            continue
        blocks = [valid_positions[i] for i in combo]
        if blocks_disjoint_correct(blocks):
            all_sets.append(tuple(sorted(blocks)))
    
    # Remove duplicates
    all_sets = list(set(all_sets))
    
    best_rank = float('inf')
    best_result = None
    
    for blocks in all_sets:
        if time.time() > deadline - 10:
            break
        
        rank = 4 * 7 + (75 - 32)  # 28 + 43 = 71
        # Actually, the number of uncovered terms depends on the blocks
        # Let's compute it properly
        
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered[i0+di][j0+dj][k0+dk] = True
        
        uncovered = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        uncovered += 1
        
        total_rank = 28 + uncovered
        
        if total_rank < best_rank:
            best_rank = total_rank
            
            U = [[0.0] * total_rank for _ in range(n * m)]
            V = [[0.0] * total_rank for _ in range(m * p)]
            W = [[0.0] * total_rank for _ in range(p * n)]
            
            col = 0
            for i0, j0, k0 in blocks:
                add_strassen_to(U, V, W, i0, j0, k0, col)
                col += 7
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if not covered[i][j][k]:
                            U[i*m+j][col] += 1
                            V[j*p+k][col] += 1
                            W[k*n+i][col] += 1
                            col += 1
            
            best_result = {"U": U, "V": V, "W": W}
    
    if best_result is None:
        # Fallback to original
        blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
        rank = 71
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i0, j0, k0 in blocks:
            add_strassen_to(U, V, W, i0, j0, k0, col)
            col += 7
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered[i0+di][j0+dj][k0+dk] = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    
    # Verify
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_ref = A @ B
    
    C_calc = np.zeros((n, p))
    for r in range(U.shape[1]):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        
        a_u = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                a_u[i,j] = u[i*m+j]
                
        b_v = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                b_v[j,k] = v[j*p+k]
                
        c_w = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                c_w[i,k] = w[k*n+i]
                
        C_calc += a_u @ b_v * c_w # This is wrong, it's outer product sum
        
    # Correct verification
    C_calc = np.zeros((n, p))
    for r in range(U.shape[1]):
        # u is vector of nm
        # v is vector of mp
        # w is vector of pn
        # Term is sum_{i,j,k} U[i,j,r] V[j,k,r] W[k,i,r] A[i,j] B[j,k]
        # = sum_{i,j} ( sum_k U[i,j,r] W[k,i,r] B[j,k] ) A[i,j] ? No.
        # It is: sum_r (u_r^T A v_r) w_r^T
        # where u_r is n x m, v_r is m x p, w_r is p x n.
        
        u_mat = U[:, r].reshape(n, m)
        v_mat = V[:, r].reshape(m, p)
        w_mat = W[:, r].reshape(p, n) # W[k*n+i] -> row k, col i? 
        # W index is k*n+i. So w_mat[k,i] = W[k*n+i].
        
        # C += (A u_mat^T ? No)
        # Term is trace(W (A U V)^T)?
        # Standard: C = sum_r (A * u_r) (v_r * B) ... no.
        # C_{ik} += sum_j A_{ij} B_{jk} W_{ki} ? No.
        # Tensor T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}.
        # C_{ik} = sum_j sum_l A_{il} B_{lk} T_{il k}? No.
        # C = A B.
        # (AB)_{ik} = sum_j A_{ij} B_{jk}.
        # We want sum_r U_{ij} V_{jk} W_{ki} = A_{ij} B_{jk}.
        # So U_{ij} V_{jk} W_{ki} is the coefficient for the monomial A_{ij} B_{jk} in the expansion?
        # No, the decomposition is for the tensor of multiplication.
        # T_{ijk} = U_{i j r} V_{j k r} W_{k i r} ? No.
        # U shape (n*m, rank). Index i*m+j.
        # V shape (m*p, rank). Index j*p+k.
        # W shape (p*n, rank). Index k*n+i.
        
        # C_{ik} = sum_j sum_r (U_{im+j, r} V_{jp+k, r} W_{kn+i, r}) A_{ij} B_{jk}
        
        a_u = np.zeros((n, m))
        a_u.flat = U[:, r] # i*m+j
        
        b_v = np.zeros((m, p))
        b_v.flat = V[:, r] # j*p+k
        
        w_w = np.zeros((p, n))
        w_w.flat = W[:, r] # k*n+i -> w_w[k,i]
        
        # C += (A * a_u.T?) No.
        # Sum over j,k: A_{ij} B_{jk} U_{ij} V_{jk} W_{ki}
        # = sum_j A_{ij} U_{ij} ( sum_k B_{jk} V_{jk} W_{ki} )
        # Let M_{j i} = sum_k V_{jk} W_{ki}
        # Then C_{ik} += sum_j A_{ij} U_{ij} M_{j i}
        # This is not a simple matrix multiply.
        
        # Simpler check:
        # Reconstruct C
        c_term = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c_term[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        C_calc += c_term
        
    err = np.max(np.abs(C_calc - C_ref))
    print(f"Rank: {U.shape[1]}, Error: {err}")
