"""K3,3-free graph on 20 vertices maximizing edges."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin before 150s kill
    
    n = 20
    
    def has_k33(adj):
        """Check if graph has K3,3. adj is list of sets."""
        # For each triple, check if they have >= 3 common neighbors
        for a, b, c in itertools.combinations(range(n), 3):
            common = adj[a] & adj[b] & adj[c]
            if len(common) >= 3:
                return True
        return False
    
    def count_k33_violations(adj):
        """Count number of K3,3 violations (triples with >=3 common neighbors)."""
        count = 0
        for a, b, c in itertools.combinations(range(n), 3):
            common = adj[a] & adj[b] & adj[c]
            if len(common) >= 3:
                count += 1
        return count
    
    def edge_count(adj):
        return sum(len(s) for s in adj) // 2
    
    # Phase 1: Greedy construction
    # Start empty, add edges in a smart order
    adj = [set() for _ in range(n)]
    
    # Try adding edges in order of increasing "conflict potential"
    # First, build a dense base using a structured approach
    # Use a bipartite-like structure but add within-part edges carefully
    
    # Strategy: Start with a complete bipartite K10,10 and remove edges to break K3,3
    # K10,10 has 100 edges but many K3,3s
    # Instead, start sparse and greedily add
    
    # Better: Use a random restart greedy approach
    best_edges = []
    best_adj = None
    best_count = 0
    
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        
        # Random initial ordering
        order = list(range(n))
        random.shuffle(order)
        
        adj = [set() for _ in range(n)]
        
        # Try all possible edges in random order, add if valid
        edges_to_try = []
        for i in range(n):
            for j in range(i+1, n):
                edges_to_try.append((i, j))
        random.shuffle(edges_to_try)
        
        for (u, v) in edges_to_try:
            if time.time() > deadline - 30:
                break
            # Check if adding (u,v) creates K3,3
            # Adding (u,v) can only create K3,3 involving u and v
            # A K3,3 is created if there exist 2 other vertices a,b such that
            # {u,v,a,b} plus one more... actually let's just check all triples
            
            # Quick check: for any pair (a,b), if a and b are both neighbors of u and v,
            # and we have a third common neighbor, we might have issues
            # Simpler: just check all triples after adding
            
            adj[u].add(v)
            adj[v].add(u)
            
            # Check if this creates a K3,3
            valid = True
            # Only need to check triples that include u or v
            for a, b, c in itertools.combinations(range(n), 3):
                if u not in (a,b,c) and v not in (a,b,c):
                    continue
                common = adj[a] & adj[b] & adj[c]
                if len(common) >= 3:
                    valid = False
                    break
            
            if not valid:
                adj[u].discard(v)
                adj[v].discard(u)
        
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_adj = [set(s) for s in adj]
            best_edges = [[u, v] for u in range(n) for v in sorted(adj[u]) if v > u]
    
    # Phase 2: Local search - edge flips
    if best_adj is not None:
        adj = [set(s) for s in best_adj]
        current_count = best_count
        
        # Precompute for efficiency
        def try_flip_remove_add(adj, remove_u, remove_v, add_u, add_v):
            """Try removing edge (remove_u, remove_v) and adding (add_u, add_v)."""
            # Remove
            adj[remove_u].discard(remove_v)
            adj[remove_v].discard(remove_u)
            
            # Check validity
            valid = True
            for a, b, c in itertools.combinations(range(n), 3):
                common = adj[a] & adj[b] & adj[c]
                if len(common) >= 3:
                    valid = False
                    break
            
            if valid:
                # Add
                adj[add_u].add(add_v)
                adj[add_v].add(add_u)
                # Check again
                for a, b, c in itertools.combinations(range(n), 3):
                    common = adj[a] & adj[b] & adj[c]
                    if len(common) >= 3:
                        valid = False
                        break
            
            if valid:
                return True
            else:
                # Revert
                adj[remove_u].add(remove_v)
                adj[remove_v].add(remove_u)
                return False
        
        # Simulated annealing style local search
        temperature = 1.0
        cooling = 0.999
        
        for iteration in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Random edge flip: remove one edge, add another
            current_edges = [(u, v) for u in range(n) for v in sorted(adj[u]) if v > u]
            if not current_edges:
                break
            
            ru, rv = random.choice(current_edges)
            # Pick random non-edge to add
            while True:
                au = random.randint(0, n-1)
                av = random.randint(0, n-1)
                if au != av and au < av and au not in adj[av]:
                    break
            
            if try_flip_remove_add(adj, ru, rv, au, av):
                current_count = edge_count(adj)
                temperature *= cooling
            else:
                temperature *= cooling
        
        final_count = edge_count(adj)
        if final_count > best_count:
            best_count = final_count
            best_adj = adj
            best_edges = [[u, v] for u in range(n) for v in sorted(adj[u]) if v > u]
    
    # Return the best result
    if best_edges is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    return best_edges

# EVOLVE-BLOCK-END
