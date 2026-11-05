import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds to be safe within 60s limit
    
    # Try smaller prime sets first
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19]
    ]
    
    # Fallback to seed solution if no better found
    best_result = list(range(1, 13))
    best_score = 9  # Score for 1..12 is 9 primes: 2,3,5,7,11,13,17,19,23
    
    def is_smooth(num, primes):
        if num <= 1:
            return True
        for p in primes:
            while num % p == 0:
                num //= p
        return num == 1
    
    def generate_smooth_numbers(max_val, primes):
        """Generate all S-smooth numbers up to max_val using a priority queue."""
        smooths = set([1])
        heap = [1]
        while heap:
            val = heapq.heappop(heap)
            if val > max_val:
                break
            if val not in smooths:
                smooths.add(val)
                for p in primes:
                    new_val = val * p
                    if new_val <= max_val:
                        heapq.heappush(heap, new_val)
        return sorted(smooths)
    
    def find_clique(nodes, adj, target_size, deadline):
        """Find a clique of target_size using backtracking with pruning."""
        if len(nodes) < target_size:
            return None
        
        # Sort by degree descending for better pruning
        nodes = sorted(nodes, key=lambda n: len(adj[n]), reverse=True)
        
        def backtrack(current_clique, candidates):
            if time.time() > deadline:
                return None
            if len(current_clique) == target_size:
                return current_clique
            if len(candidates) < target_size - len(current_clique):
                return None
            
            for i in range(len(candidates)):
                node = candidates[i]
                new_clique = current_clique + [node]
                # New candidates: those after i that are adjacent to node
                new_candidates = [c for c in candidates[i+1:] if c in adj[node]]
                res = backtrack(new_clique, new_candidates)
                if res:
                    return res
            return None
        
        return backtrack([], nodes)
    
    for primes in prime_sets:
        if time.time() > deadline:
            break
        
        smooths = generate_smooth_numbers(10000, primes)
        if len(smooths) < 12:
            continue
        
        # Build adjacency: edge between u,v if u+v is S-smooth
        n = len(smooths)
        adj = {s: set() for s in smooths}
        
        # O(N^2) edge building
        for i in range(n):
            u = smooths[i]
            for j in range(i + 1, n):
                v = smooths[j]
                s = u + v
                if is_smooth(s, primes):
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Filter candidates: must have degree >= 11 (need 11 other nodes in clique)
        candidates = [s for s in smooths if len(adj[s]) >= 11]
        
        if len(candidates) >= 12:
            clique = find_clique(candidates, adj, 12, deadline)
            if clique:
                # Verify the clique
                valid = True
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        s = clique[i] + clique[j]
                        if not is_smooth(s, primes):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    if len(primes) < best_score:
                        best_result = clique
                        best_score = len(primes)
                    break  # Found best for this family size; since we iterate smallest first, this is optimal
    
    return best_result
