import time
import numpy as np
from itertools import combinations

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition for given j-pairs and k-pairs."""
        num_blocks = len(j_pairs) * len(k_pairs)
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen terms for 2x2x2 block
                # P1 = (A0j0+A1j1)*(Bj0k0+Bj1k1)
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P2 = (A1j0+A1j1)*Bj0k0
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                # P3 = A0j0*(Bj0k1-Bj1k1)
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P4 = A1j1*(Bj1k0-Bj0k0)
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                # P5 = (A0j0+A0j1)*Bj1k1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                # P6 = (A1j0-A0j0)*(Bj0k0+Bj0k1)
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                # P7 = (A0j1-A1j1)*(Bj1k0+Bj1k1)
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        
        # Remaining j: standard products
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        
        return U, V, W, rank

    def try_eliminate_terms(U, V, W, rank):
        """Try to eliminate redundant terms by checking each term individually."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        cur_rank = rank
        
        changed = True
        while changed and time_left() > 5:
            changed = False
            for term_idx in range(cur_rank - 1, -1, -1):
                if time_left() < 5:
                    break
                
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][term_idx] = 0
                for j in range(m*p):
                    V_new[j][term_idx] = 0
                for k in range(p*n):
                    W_new[k][term_idx] = 0
                
                if verify(U_new, V_new, W_new, cur_rank):
                    U_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in W_list]
                    cur_rank -= 1
                    changed = True
                    break
        
        return U_list, V_list, W_list, cur_rank

    def try_pair_elimination(U, V, W, rank):
        """Try to eliminate pairs of terms that cancel each other."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        cur_rank = rank
        
        # Check if removing two terms still gives valid decomposition
        for term1 in range(cur_rank - 1):
            if time_left() < 10:
                return U_list, V_list, W_list, cur_rank
            for term2 in range(term1 + 1, cur_rank):
                if time_left() < 5:
                    return U_list, V_list, W_list, cur_rank
                
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][term1] = 0
                    U_new[i][term2] = 0
                for j in range(m*p):
                    V_new[j][term1] = 0
                    V_new[j][term2] = 0
                for k in range(p*n):
                    W_new[k][term1] = 0
                    W_new[k][term2] = 0
                
                if verify(U_new, V_new, W_new, cur_rank):
                    # Remove both terms
                    keep = [j for j in range(cur_rank) if j != term1 and j != term2]
                    U_list = [[row[j] for j in keep] for row in U_list]
                    V_list = [[row[j] for j in keep] for row in V_list]
                    W_list = [[row[j] for j in keep] for row in W_list]
                    cur_rank -= 2
                    return U_list, V_list, W_list, cur_rank
        
        return U_list, V_list, W_list, cur_rank

    # Generate all valid pairings for j (5 elements -> 2 pairs + 1 remaining)
    def generate_pairings(num_elements):
        """Generate all ways to partition into pairs and singles."""
        results = []
        
        def helper(remaining, current_pairs, current_singles):
            if not remaining:
                results.append((current_pairs[:], current_singles[:]))
                return
            
            first = remaining[0]
            rest = remaining[1:]
            
            # Make first a single
            helper(rest, current_pairs, current_singles + [first])
            
            # Pair first with each other element
            for i, other in enumerate(rest):
                new_remaining = rest[:i] + rest[i+1:]
                helper(new_remaining, current_pairs + [(first, other)], current_singles)
        
        helper(list(range(num_elements)), [], [])
        return results

    # Generate all valid pairings for k (6 elements -> 3 pairs)
    def generate_perfect_matchings(num_elements):
        """Generate all perfect matchings (pairings) of elements."""
        results = []
        
        def helper(remaining, current_pairs):
            if not remaining:
                results.append(current_pairs[:])
                return
            
            first = remaining[0]
            rest = remaining[1:]
            
            for i, other in enumerate(rest):
                new_remaining = rest[:i] + rest[i+1:]
                helper(new_remaining, current_pairs + [(first, other)])
        
        helper(list(range(num_elements)), [])
        return results

    # Precompute all valid configurations
    j_pairings_with_singles = generate_pairings(m)
    k_perfect_matchings = generate_perfect_matchings(p)
    
    # Filter: we want exactly 2 j-pairs (4 j's used, 1 remaining)
    valid_j_configs = []
    for pairs, singles in j_pairings_with_singles:
        if len(pairs) == 2 and len(singles) == 1:
            valid_j_configs.append((pairs, singles))
    
    valid_k_configs = k_perfect_matchings  # All have 3 pairs
    
    # Sort by expected rank (lower is better)
    def config_rank(j_pairs, k_pairs):
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = m - len(used_j)
        return 7 * len(j_pairs) * len(k_pairs) + 2 * remaining_j * p
    
    # Generate all configurations and sort by rank
    all_configs = []
    for j_pairs, _ in valid_j_configs:
        for k_pairs in valid_k_configs:
            rank = config_rank(j_pairs, k_pairs)
            all_configs.append((rank, j_pairs, k_pairs))
    
    all_configs.sort(key=lambda x: x[0])
    
    best_result = None
    best_rank = float('inf')
    
    # Try configurations in order of increasing rank
    for rank_est, j_pairs, k_pairs in all_configs:
        if time_left() < 15:
            break
        
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        # Try single term elimination
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        # Try pair elimination if time permits
        if time_left() > 20 and new_rank > 50:
            U_list, V_list, W_list, new_rank = try_pair_elimination(U_list, V_list, W_list, new_rank)
        
        # Try single elimination again after pair elimination
        if time_left() > 10 and new_rank > 50:
            U_list, V_list, W_list, new_rank = try_eliminate_terms(U_list, V_list, W_list, new_rank)
        
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
        
        # If we've found a good result, we can stop early
        if best_rank <= 50:
            break
    
    # Fallback to schoolbook if nothing worked
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
