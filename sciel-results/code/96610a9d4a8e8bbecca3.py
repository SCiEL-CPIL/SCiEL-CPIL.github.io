"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 140  # Safety margin
    
    def validate(U, V, W, rank):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def strassen_2x2x2(U, V, W, col, i0, i1, j0, j1, k0, k1):
        """Add Strassen decomposition for 2x2x2 block. Returns new col."""
        # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        U[i1*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+i1, col] += 1; W[k1*n+i1, col] -= 1
        col += 1
        # P3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        U[i0*m+j0, col] += 1
        V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
        W[k1*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        U[i1*m+j1, col] += 1
        V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+i0, col] += 1; W[k0*n+i1, col] += 1
        col += 1
        # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        U[i0*m+j0, col] += 1; U[i0*m+j1, col] += 1
        V[j1*p+k1, col] += 1
        W[k0*n+i0, col] -= 1; W[k1*n+i0, col] += 1
        col += 1
        # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        U[i1*m+j0, col] += 1; U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
        W[k1*n+i1, col] += 1
        col += 1
        # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        U[i0*m+j1, col] += 1; U[i1*m+j1, col] -= 1
        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1
        col += 1
        return col
    
    def schoolbook_block(U, V, W, col, i_range, j_range, k_range):
        """Add schoolbook terms for a block. Returns new col."""
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i*m+j, col] += 1
                    V[j*p+k, col] += 1
                    W[k*n+i, col] += 1
                    col += 1
        return col
    
    def try_strassen_split():
        rank = 54
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for j0, j1 in [(0,1), (2,3)]:
            for k0, k1 in [(0,1), (2,3), (4,5)]:
                col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
        col = schoolbook_block(U, V, W, col, [0,1], [4], range(p))
        return U, V, W, rank
    
    U, V, W, rank = try_strassen_split()
    if not validate(U, V, W, rank):
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = schoolbook_block(U, V, W, 0, range(n), range(m), range(p))
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Strategy 2: Systematic search over block partitions with Strassen
    if time.time() < deadline - 30:
        def find_best_partition():
            """Search over ways to partition j and k dimensions, using Strassen where possible."""
            best_rank_found = best_rank
            best_result = None
            
            # Try all ways to split m=5 into groups of size 1 or 2
            # and p=6 into groups of size 1, 2, or 3
            # For each (j_group, k_group) pair, use Strassen if both sizes are 2, else schoolbook
            
            import itertools
            
            def gen_splits(total, max_size=2):
                """Generate all ways to split 'total' into parts of size 1 or max_size."""
                if total == 0:
                    yield []
                    return
                for size in range(1, min(max_size, total) + 1):
                    for rest in gen_splits(total - size, max_size):
                        yield [size] + rest
            
            def compute_rank(j_split, k_split):
                """Compute the rank for a given partition."""
                rank = 0
                for js in j_split:
                    for ks in k_split:
                        if js == 2 and ks == 2:
                            rank += 7  # Strassen
                        else:
                            rank += n * js * ks  # schoolbook
                return rank
            
            def build_decomposition(j_split, k_split):
                """Build the U, V, W matrices for a given partition."""
                rank = compute_rank(j_split, k_split)
                if rank >= best_rank_found:
                    return None
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                col = 0
                j_offset = 0
                for js in j_split:
                    j_indices = list(range(j_offset, j_offset + js))
                    k_offset = 0
                    for ks in k_split:
                        k_indices = list(range(k_offset, k_offset + ks))
                        if js == 2 and ks == 2:
                            col = strassen_2x2x2(U, V, W, col, 0, 1, j_indices[0], j_indices[1], k_indices[0], k_indices[1])
                        else:
                            col = schoolbook_block(U, V, W, col, [0, 1], j_indices, k_indices)
                        k_offset += ks
                    j_offset += js
                return U, V, W, rank
            
            # Try all j splits (m=5) with parts of size 1 or 2
            j_splits = list(gen_splits(5, max_size=2))
            # Try all k splits (p=6) with parts of size 1, 2, or 3
            k_splits = list(gen_splits(6, max_size=3))
            
            # Sort by estimated rank to try best first
            candidates = []
            for js in j_splits:
                for ks in k_splits:
                    r = compute_rank(js, ks)
                    candidates.append((r, js, ks))
            candidates.sort()
            
            for r, js, ks in candidates:
                if r >= best_rank_found:
                    break
                if time.time() > deadline - 10:
                    break
                result = build_decomposition(js, ks)
                if result is None:
                    continue
                U, V, W, rank = result
                if validate(U, V, W, rank):
                    best_rank_found = rank
                    best_result = (U, V, W, rank)
                    break  # Found the best possible for this candidate
            
            return best_result
        
        result = find_best_partition()
        if result is not None:
            U2, V2, W2, rank2 = result
            if rank2 < best_rank:
                best_U, best_V, best_W, best_rank = U2, V2, W2, rank2
    
    # Strategy 3: Try to find rank < 54 by using non-rectangular Strassen-like decompositions
    if time.time() < deadline - 15:
        # Try specific hand-crafted decompositions that might beat 54
        # Key idea: for 2x5x6, we can try to share factors more cleverly
        
        def try_rank_50():
            """Attempt a rank-50 decomposition using mixed Strassen/schoolbook."""
            # Partition: j={0,1}, {2,3}, {4}; k={0,1}, {2,3}, {4,5}
            # 2x2x2 Strassen: 7 each
            # 2x1x2 schoolbook: 4 each  
            # 1x2x2 schoolbook: 4 each
            # 1x1x2 schoolbook: 2 each
            # 1x2x1 schoolbook: 2 each
            # 1x1x1 schoolbook: 1 each
            # 
            # j01-k01: 7, j01-k23: 7, j01-k45: 7
            # j23-k01: 7, j23-k23: 7, j23-k45: 7
            # j4-k01: 4, j4-k23: 4, j4-k45: 4
            # Total: 7*6 + 4*3 = 42+12 = 54
            # Same as before. Need a different approach.
            return None
        
        def try_rank_48():
            """Try a more aggressive decomposition."""
            # What if we can do 2x2x3 with rank 9 instead of 12?
            # Or 2x3x2 with rank 9?
            # Unknown if such decompositions exist.
            # For now, stick with known Strassen.
            return None
        
        # Try a different partition: m=2+3, p=3+3
        # j={0,1}, {2,3,4}; k={0,1,2}, {3,4,5}
        # 2x2x3: schoolbook=12 (no Strassen for 2x2x3)
        # 2x2x3: schoolbook=12
        # 2x3x3: schoolbook=18
        # 2x3x3: schoolbook=18
        # Total: 60 (worse)
        
        # Try m=1+2+2, p=3+3
        # j={0}, {1,2}, {3,4}; k={0,1,2}, {3,4,5}
        # 2x1x3: 6, 2x1x3: 6
        # 2x2x3: 12, 2x2x3: 12
        # 2x2x3: 12, 2x2x3: 12
        # Total: 60 (worse)
        
        pass  # No improvement found in this strategy
    
    # Final validation
    if not validate(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        best_rank = n*m*p
        best_U = np.zeros((n*m, best_rank))
        best_V = np.zeros((m*p, best_rank))
        best_W = np.zeros((p*n, best_rank))
        col = schoolbook_block(best_U, best_V, best_W, 0, range(n), range(m), range(p))
    
    # Convert to lists for the return interface
    U_list = [[int(best_U[i, c]) if abs(best_U[i, c] - round(best_U[i, c])) < 1e-9 else float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[int(best_V[i, c]) if abs(best_V[i, c] - round(best_V[i, c])) < 1e-9 else float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[int(best_W[i, c]) if abs(best_W[i, c] - round(best_W[i, c])) < 1e-9 else float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
