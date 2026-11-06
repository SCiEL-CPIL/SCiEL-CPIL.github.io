import random
import time
import itertools
import numpy as np

def construct():
    """
    Constructs a graph on n vertices with no regular induced subgraph on k >= 6 vertices.
    Uses vectorized NumPy-based local search with degree-distribution pruning.
    """
    start_time = time.time()
    time_limit = 145.0  # 5s buffer for 150s limit

    def adj_to_matrix(adj, n):
        """Convert adjacency list (bitmasks) to numpy boolean matrix."""
        mat = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                mat[i, j] = 1
                mask ^= lsb
        return mat

    def matrix_to_adj(mat, n):
        """Convert numpy matrix back to adjacency list (bitmasks)."""
        adj = [0] * n
        for i in range(n):
            mask = 0
            row = mat[i]
            for j in range(n):
                if row[j]:
                    mask |= (1 << j)
            adj[i] = mask
        return adj

    def check_violations_fast(mat, n, max_violations=50):
        """
        Check for regular induced subgraphs of size k >= 6 using vectorized operations.
        Returns list of (k, vertex_indices) tuples for violations found.
        Uses degree-distribution pruning: for a subset to be regular, all induced degrees must be equal.
        """
        violations = []
        
        # Precompute all-pairs adjacency for fast degree computation
        # For a subset S, the induced degree of vertex v in S is sum(mat[v, S])
        
        for k in range(6, n + 1):
            found_in_k = 0
            for combo in itertools.combinations(range(n), k):
                # Convert combo to boolean mask
                S = np.zeros(n, dtype=np.uint8)
                for v in combo:
                    S[v] = 1
                
                # Compute induced degrees for all vertices in S
                # deg[v] = mat[v, :].dot(S) for v in S
                degs = mat[combo].dot(S)
                
                # Check if all degrees are equal
                d0 = degs[0]
                if np.all(degs == d0):
                    violations.append((k, combo))
                    found_in_k += 1
                    if found_in_k >= max_violations:
                        return violations
            
            if found_in_k == 0 and k > 6:
                # If no violations of size k, larger subsets are less likely but still possible
                # Continue checking
                pass
        
        return violations

    def count_violations_quick(mat, n):
        """
        Quick count of violations - only checks k=6 first for speed.
        Returns (count, sample_violations).
        """
        count = 0
        samples = []
        for combo in itertools.combinations(range(n), 6):
            S = np.zeros(n, dtype=np.uint8)
            for v in combo:
                S[v] = 1
            degs = mat[combo].dot(S)
            if np.all(degs == degs[0]):
                count += 1
                if len(samples) < 20:
                    samples.append(combo)
                if count > 50:
                    break
        return count, samples

    def local_search_vectorized(n, initial_adj, deadline):
        """
        Local search using vectorized degree computations.
        Returns (success, final_adj).
        """
        mat = adj_to_matrix(initial_adj, n)
        current_count, _ = count_violations_quick(mat, n)
        
        if current_count == 0:
            # Verify fully
            if not check_violations_fast(mat, n):
                return True, matrix_to_adj(mat, n)
        
        best_count = current_count
        best_mat = mat.copy()
        no_improve = 0
        
        while time.time() < deadline:
            if no_improve > 200:
                break
            
            # Get some violation samples
            _, samples = count_violations_quick(mat, n)
            if not samples:
                # No k=6 violations, do full check
                full_viol = check_violations_fast(mat, n)
                if not full_viol:
                    return True, matrix_to_adj(mat, n)
                # Use a full violation
                k, combo = full_viol[0]
                samples = [combo]
            
            # Pick a random violation
            combo = random.choice(samples)
            
            # Strategy: flip an edge within the violation subgraph
            # This changes degrees of exactly 2 vertices in the subgraph
            if len(combo) < 2:
                continue
            
            # Try multiple random edge flips within the violation
            best_flip = None
            best_new_count = best_count
            
            for _ in range(3):
                u, v = random.sample(combo, 2)
                
                # Save state
                old_uv = mat[u, v]
                
                # Flip
                mat[u, v] = 1 - old_uv
                mat[v, u] = 1 - old_uv
                
                # Evaluate
                new_count, _ = count_violations_quick(mat, n)
                
                if new_count < best_new_count:
                    best_new_count = new_count
                    best_flip = (u, v, old_uv)
                
                # Revert for next trial
                mat[u, v] = old_uv
                mat[v, u] = old_uv
            
            if best_flip is not None:
                u, v, old_uv = best_flip
                mat[u, v] = 1 - old_uv
                mat[v, u] = 1 - old_uv
                
                if best_new_count < best_count:
                    best_count = best_new_count
                    best_mat = mat.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                
                if best_count == 0:
                    # Full verification
                    if not check_violations_fast(mat, n):
                        return True, matrix_to_adj(mat, n)
            else:
                no_improve += 1
        
        return False, matrix_to_adj(best_mat, n)

    def try_build(n, deadline):
        """Try to build a valid graph on n vertices."""
        max_restarts = 15 if n <= 20 else 8
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize: random graph with edge probability ~0.5
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            success, final_adj = local_search_vectorized(n, adj, deadline)
            
            if success:
                return final_adj
        
        return None

    best_result = {'n': 1, 'edges': []}
    
    # Try from highest n down
    for target_n in [22, 21, 20]:
        if time.time() > time_limit - 8:
            break
        
        result_adj = try_build(target_n, time_limit - 3)
        
        if result_adj is not None:
            edges = []
            n = target_n
            for i in range(n):
                for j in range(i + 1, n):
                    if (result_adj[i] >> j) & 1:
                        edges.append([i, j])
            best_result = {'n': target_n, 'edges': edges}
            break
    
    # Fallback: if we only got n=1, try a simpler approach
    if best_result['n'] < 5 and time.time() < time_limit - 5:
        # Greedy: build up vertex by vertex
        current_n = 1
        current_adj = [0]
        for target_n in range(2, 16):
            if time.time() > time_limit - 2:
                break
            found = False
            for _ in range(50):
                new_adj = current_adj[:] + [0]
                for i in range(current_n):
                    if random.random() < 0.5:
                        new_adj[i] |= (1 << current_n)
                        new_adj[current_n] |= (1 << i)
                # Quick check: just verify no k=6 regular subgraphs
                mat = adj_to_matrix(new_adj, target_n)
                cnt, _ = count_violations_quick(mat, target_n)
                if cnt == 0:
                    # Full check
                    if not check_violations_fast(mat, target_n):
                        current_adj = new_adj
                        current_n = target_n
                        found = True
                        break
            if not found:
                break
        
        if current_n > best_result['n']:
            edges = [[i, j] for i in range(current_n) for j in range(i + 1, current_n) if (current_adj[i] >> j) & 1]
            best_result = {'n': current_n, 'edges': edges}

    return best_result

if __name__ == "__main__":
    construct()
