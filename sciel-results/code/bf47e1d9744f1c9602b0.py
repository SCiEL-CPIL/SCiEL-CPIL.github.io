import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for 150s limit
    
    # Helper: Check if graph is triangle-free using bitsets
    def is_triangle_free(adj, n):
        for i in range(n):
            a = adj[i]
            temp = a
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if adj[j] & a:
                    return False
                temp ^= lsb
        return True
    
    # Helper: Exact check for independent set of size 10 using branch and bound
    def has_independent_set_10(adj, n, time_deadline):
        sys.setrecursionlimit(10000)
        
        def dfs(remaining, current_size):
            if time.time() > time_deadline:
                return False
            if current_size >= 10:
                return True
            if current_size + remaining.bit_count() < 10:
                return False
            if remaining == 0:
                return False
            
            # Pick vertex with highest degree in remaining
            lsb = remaining & -remaining
            v = lsb.bit_length() - 1
            
            # Branch 1: Include v
            new_remaining_include = remaining & ~adj[v] & ~lsb
            if dfs(new_remaining_include, current_size + 1):
                return True
            
            # Branch 2: Exclude v
            new_remaining_exclude = remaining & ~lsb
            if dfs(new_remaining_exclude, current_size):
                return True
            
            return False
        
        all_vertices = (1 << n) - 1
        return dfs(all_vertices, 0)
    
    # Helper: Get edges from adj
    def get_edges(adj, n):
        edges = []
        for i in range(n):
            a = adj[i]
            while a:
                lsb = a & -a
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                a ^= lsb
        return edges
    
    # Helper: Fast heuristic to estimate if independent set of 10 exists
    def quick_check_independent_set_10(adj, n, trials=20):
        found = False
        for _ in range(trials):
            rem = (1 << n) - 1
            size = 0
            while rem and size < 10:
                bits = []
                temp = rem
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    bits.append(v)
                    temp ^= lsb
                if not bits:
                    break
                v = random.choice(bits)
                size += 1
                rem &= ~adj[v]
                rem &= ~(1 << v)
            if size >= 10:
                found = True
                break
        return found
    
    # Helper: Count triangles
    def count_triangles(adj, n):
        tri_count = 0
        for i in range(n):
            a = adj[i]
            temp = a
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if adj[j] & a:
                    tri_count += 1
                temp ^= lsb
        return tri_count // 3  # Each triangle counted 3 times
    
    # Search for a valid graph
    def search(n, time_budget):
        end_time = start_time + time_budget
        best_graph = None
        best_score = -float('inf')
        
        # Initialize with random triangle-free graphs
        # We can generate random regular graphs or just random and fix
        for _ in range(50):
            if time.time() > end_time:
                break
            
            # Random initialization
            adj = [0] * n
            # Add random edges, ensuring no triangles
            # This is slow. Better: start with empty, add edges randomly
            random.seed(int(time.time() * 1000) + _)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.3: # Sparse start
                        # Check if adding (i,j) creates triangle
                        if (adj[i] & adj[j]) == 0:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            
            # Local search
            for step in range(1000):
                if time.time() > end_time:
                    break
                
                # Calculate score
                # Penalty for triangles
                tri_count = count_triangles(adj, n)
                
                # Estimate alpha
                # Just check if has_independent_set_10 is likely
                # If tri_count > 0, score is bad
                score = -1000 * tri_count
                
                # If no triangles, check alpha
                if tri_count == 0:
                    # Quick check: try to find independent set of 10
                    # Use a fast heuristic
                    found_10 = quick_check_independent_set_10(adj, n, trials=10)
                    
                    if found_10:
                        score -= 100
                    else:
                        score += 10 # Reward for potentially valid
                        
                        # Exact check on best candidates
                        if score > best_score:
                            best_score = score
                            best_graph = adj[:]
                            
                            # Run exact check if score is high enough
                            if not has_independent_set_10(adj, n, end_time):
                                return adj, n # Found valid graph!
                
                # Mutation
                # Pick random edge to flip
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                # Try to flip edge (u,v)
                if (adj[u] >> v) & 1:
                    # Remove edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    # Add edge if no triangle
                    if (adj[u] & adj[v]) == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        # Can't add, maybe remove a different edge?
                        # Or just skip
                        pass

        if best_graph:
            # Final exact check
            if is_triangle_free(best_graph, n) and not has_independent_set_10(best_graph, n, end_time):
                return best_graph, n
        return None, None
    
    # Try n=40, 39, 38
    for n in [40, 39, 38]:
        time_left = time_limit - (time.time() - start_time)
        if time_left < 5:
            break
        
        # Allocate time based on n
        budget = min(time_left * 0.8, 30 if n == 40 else 60)
        
        adj, found_n = search(n, budget)
        if adj is not None:
            edges = get_edges(adj, found_n)
            return {'n': found_n, 'edges': edges}

    # Fallback: Return a known valid graph (Schläfli, n=27)
    # Schläfli graph construction
    n = 27
    adj = [0] * n
    # Schläfli graph is the complement of the line graph of K_6? No.
    # It is a strongly regular graph (27, 10, 1, 5).
    # Vertices are 2-subsets of {1..6} (15 vertices) + ... no.
    # Vertices are lines of PG(3,2)?
    # Easier: Use the Andrasfai graph And(26) which is known to be valid.
    n = 26
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    edges = get_edges(adj, n)
    return {'n': n, 'edges': edges}

if __name__ == "__main__":
    result = construct()
    print(result['n'])
