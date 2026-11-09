import random
import time
import itertools
from collections import Counter

def construct():
    """
    Construct a graph on n vertices with no regular induced subgraph on 6 or more vertices.
    Maximize n.
    """
    
    def is_regular_subgraph(adj, subset):
        """
        Check if the induced subgraph on 'subset' is regular.
        adj is a list of integer bitmasks.
        subset is a list of vertex indices.
        """
        if len(subset) < 6:
            return False
        
        # Map subset vertices to local indices 0..k-1 for faster mask ops?
        # Actually, just compute degrees.
        k = len(subset)
        subset_set = set(subset)
        mask = 0
        for v in subset:
            mask |= (1 << v)
            
        degrees = []
        for v in subset:
            # Count neighbors of v that are in subset
            deg = (adj[v] & mask).bit_count()
            degrees.append(deg)
        
        # Check if all degrees are the same
        if not degrees:
            return False
        first_deg = degrees[0]
        for d in degrees[1:]:
            if d != first_deg:
                return False
        return True

    def find_regular_subgraph(adj, n, min_size=6, max_size=None):
        """
        Find any regular induced subgraph of size >= min_size.
        Returns (subset, size) or (None, 0).
        """
        if max_size is None:
            max_size = n
            
        # Check smaller sizes first as they are more likely to exist and cheaper?
        # Or check size 6 first as it's the most numerous.
        # Let's iterate size from min_size to max_size.
        for size in range(min_size, max_size + 1):
            for subset in itertools.combinations(range(n), size):
                if is_regular_subgraph(adj, list(subset)):
                    return list(subset), size
        return None, 0

    def solve_for_n(n, time_limit=50):
        """
        Try to find a valid graph for specific n using local search.
        """
        if n < 6:
            # Trivially valid if n < 6? No, "No set of 6 or more". 
            # If n < 6, there are no such sets. So any graph works.
            # But we want to maximize n.
            return {'n': n, 'edges': []}
            
        # Initialize random graph
        adj = [0] * n
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.append([i, j])
        
        start_time = time.time()
        max_iter = 50000
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
                
            # Find a bad subgraph
            bad_subset, bad_size = find_regular_subgraph(adj, n, min_size=6)
            
            if bad_subset is None:
                # Verify strictly one more time? find_regular_subgraph does full check.
                return {'n': n, 'edges': edges}
            
            # We found a regular subgraph. Try to break it.
            # Pick a random edge within the bad subset to flip.
            # Flipping an edge inside the subset changes degrees of vertices in the subset.
            # Flipping an edge outside might also help if we consider larger structures, 
            # but breaking the regularity of the specific subset is the immediate goal.
            
            # Select a random pair of vertices in the bad_subset
            u, v = random.sample(bad_subset, 2)
            
            # Flip the edge (u, v)
            if adj[u] & (1 << v):
                # Edge exists, remove it
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                edges.remove([u, v]) if [u, v] in edges else edges.remove([v, u])
                # Note: edges list maintenance is slow. 
                # Let's reconstruct edges at the end if we succeed.
                # For now, just track adj. We'll rebuild edges at return.
            else:
                # Edge doesn't exist, add it
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            # Optimization: Rebuilding edges list every step is expensive.
            # We can just keep adj and generate edges at the end.
            
        # If loop finishes without success, try a few more aggressive moves or restart?
        # For now, return None to signal failure for this n.
        return None

    # Try n from 20 down to 6
    for n in range(20, 5, -1):
        # Try multiple restarts for each n
        found = False
        for attempt in range(5):
            res = solve_for_n(n, time_limit=10)
            if res:
                # Double check the result
                # Reconstruct adj from edges in res
                adj_check = [0] * n
                for u, v in res['edges']:
                    adj_check[u] |= (1 << v)
                    adj_check[v] |= (1 << u)
                
                valid = True
                # Check all subsets of size >= 6
                # This is the expensive part. 
                # For n=20, this is ~200k checks. Each check is O(n^2) bit ops. 
                # Should be under 1 second in Python if optimized?
                # itertools.combinations is slow.
                
                # Let's do a quick sanity check on size 6 only during search, 
                # and full check here.
                
                for size in range(6, n + 1):
                    for subset in itertools.combinations(range(n), size):
                        if is_regular_subgraph(adj_check, list(subset)):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    return res
        # If not found for this n, try smaller n

    # Fallback: return a trivial valid graph (e.g., n=5)
    # But the task says maximize n. n=5 is valid (no subset of size 6).
    # The seed returns n=5. We should at least do better.
    # Let's try n=6. A 6-vertex graph is regular iff it's regular.
    # We just need a 6-vertex graph that is NOT regular.
    # E.g., a path P6.
    return {'n': 6, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]}

# EVOLVE-BLOCK-START
def construct():
    # Placeholder to match structure, real logic in helper or inline?
    # The task requires keeping the function name construct().
    # I will implement the logic directly inside construct() or call the helper.
    # To keep it clean, I'll paste the logic here.
    
    import random
    import time
    import itertools

    def is_regular_subgraph(adj, subset):
        k = len(subset)
        if k < 6:
            return False
        mask = 0
        for v in subset:
            mask |= (1 << v)
        degrees = []
        for v in subset:
            deg = (adj[v] & mask).bit_count()
            degrees.append(deg)
        if not degrees:
            return False
        first_deg = degrees[0]
        for d in degrees[1:]:
            if d != first_deg:
                return False
        return True

    def find_regular_subgraph(adj, n, min_size=6, max_size=None):
        if max_size is None:
            max_size = n
        for size in range(min_size, max_size + 1):
            for subset in itertools.combinations(range(n), size):
                if is_regular_subgraph(adj, list(subset)):
                    return list(subset)
        return None

    def solve_for_n(n, time_limit=10):
        if n < 6:
            return {'n': n, 'edges': []}
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        start_time = time.time()
        max_iter = 10000
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            bad_subset = find_regular_subgraph(adj, n, min_size=6)
            
            if bad_subset is None:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Pick random edge in bad_subset to flip
            u, v = random.sample(bad_subset, 2)
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return None

    # Try from 20 down to 6
    for n in range(20, 5, -1):
        for attempt in range(3):
            res = solve_for_n(n, time_limit=15)
            if res:
                # Final verification
                adj_check = [0] * n
                for u, v in res['edges']:
                    adj_check[u] |= (1 << v)
                    adj_check[v] |= (1 << u)
                
                valid = True
                for size in range(6, n + 1):
                    for subset in itertools.combinations(range(n), size):
                        if is_regular_subgraph(adj_check, list(subset)):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return res

    # Fallback n=6
    return {'n': 6, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]}

# EVOLVE-BLOCK-END
