import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    def make_paley(q):
        """Create Paley graph on q vertices."""
        squares = set(x * x % q for x in range(1, q))
        adj = [0] * q
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.append([i, j])
        return adj, edges
    
    def has_k4(adj, n):
        """Check if graph contains K4 using bit operations."""
        for i in range(n):
            ni = adj[i]
            for j_bit in ni:
                j = j_bit.bit_length() - 1
                if j <= i:
                    continue
                nij = ni & adj[j]
                # Check if nij contains an edge
                remaining = nij
                while remaining:
                    k_bit = remaining & (-remaining)
                    k = k_bit.bit_length() - 1
                    remaining ^= k_bit
                    if k <= j:
                        continue
                    nijk = nij & adj[k]
                    if nijk:
                        for l_bit in nijk:
                            l = l_bit.bit_length() - 1
                            if l > k:
                                return True
        return False
    
    def has_indep_6(adj, n):
        """Check if graph contains independent set of size 6."""
        # Complement adjacency: ~adj[i] & mask
        mask = (1 << n) - 1
        
        def count_max_indep(adj, n):
            # Branch and bound on complement (clique in complement)
            best = [0]
            comp_adj = [mask ^ adj[i] & (mask ^ (1 << i)) for i in range(n)]
            
            def bnb(candidates, size):
                if size >= 6:
                    return True
                if not candidates:
                    return False
                # Prune
                if size + candidates.bit_count() < 6:
                    return False
                # Pick a vertex
                v_bit = candidates & (-candidates)
                v = v_bit.bit_length() - 1
                remaining = candidates ^ v_bit
                # Include v
                if bnb(remaining & comp_adj[v], size + 1):
                    return True
                # Exclude v
                if bnb(remaining, size):
                    return True
                return False
            
            return bnb(mask, 0)
        
        return count_max_indep(adj, n)
    
    def is_valid(adj, n):
        return not has_k4(adj, n) and not has_indep_6(adj, n)
    
    def fitness(adj, n):
        """Lower is better. 0 means valid."""
        score = 0
        if has_k4(adj, n):
            score += 1000
        if has_indep_6(adj, n):
            score += 1000
        return score
    
    def flip_edge(adj, i, j):
        """Flip edge between i and j, return new adjacency list."""
        new_adj = adj[:]
        if adj[i] & (1 << j):
            new_adj[i] &= ~(1 << j)
            new_adj[j] &= ~(1 << i)
        else:
            new_adj[i] |= (1 << j)
            new_adj[j] |= (1 << i)
        return new_adj
    
    def local_search(adj, n, max_iters, deadline):
        """Simulated annealing / hill climbing to find valid graph."""
        current = adj[:]
        current_fit = fitness(current, n)
        best = current[:]
        best_fit = current_fit
        temp = 1.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if current_fit == 0:
                best = current[:]
                best_fit = 0
                return best, 0
            
            # Pick random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            new_adj = flip_edge(current, i, j)
            new_fit = fitness(new_adj, n)
            
            delta = new_fit - current_fit
            if delta < 0 or random.random() < (1.0 / (1.0 + delta * 0.1)) * temp:
                current = new_adj
                current_fit = new_fit
                if current_fit < best_fit:
                    best = current[:]
                    best_fit = current_fit
            
            temp *= cooling
        
        return best, best_fit
    
    def try_extend(adj_base, n_base, target_n, deadline):
        """Try to extend base graph to target_n vertices."""
        if n_base >= target_n:
            return None
        
        # Start with base graph plus random connections for new vertices
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Create initial guess: new vertices connect to random subset
            adj = adj_base[:]
            for new_v in range(n_base, target_n):
                adj.append(0)
                # Connect to random ~50% of existing vertices
                for old_v in range(n_base):
                    if random.random() < 0.5:
                        adj[old_v] |= (1 << new_v)
                        adj[new_v] |= (1 << old_v)
                # Connect to other new vertices randomly
                for other_new in range(n_base, new_v):
                    if random.random() < 0.5:
                        adj[other_new] |= (1 << new_v)
                        adj[new_v] |= (1 << other_new)
            
            # Local search
            improved, fit = local_search(adj, target_n, 3000, deadline - 5.0)
            if fit == 0:
                return improved
        
        return None
    
    # Start with Paley(29) - known to be K4-free and alpha-free(6)
    q = 29
    adj, edges = make_paley(q)
    n = q
    
    # Verify
    if not is_valid(adj, n):
        # Try smaller primes
        for q in [17, 13, 11]:
            adj, edges = make_paley(q)
            n = q
            if is_valid(adj, n):
                break
    
    best_n = n
    best_adj = adj[:]
    
    # Try to extend
    for target in range(n + 1, 36):
        if time.time() > deadline - 10.0:
            break
        
        result = try_extend(best_adj, best_n, target, deadline)
        if result is not None:
            best_adj = result
            best_n = target
        else:
            break
    
    # Build edge list from best_adj
    final_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                final_edges.append([i, j])
    
    return {'n': best_n, 'edges': final_edges}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
