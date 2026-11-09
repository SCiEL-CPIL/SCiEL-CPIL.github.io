import random
import time

def construct():
    n = 20
    # Split into two sets A (0-9) and B (10-19)
    A = list(range(10))
    B = list(range(10, 20))
    
    # Adjacency as bitsets for B vertices
    # adj[u] is a bitset of neighbors in B
    adj = [0] * 10
    
    # Initialize with a random graph or structured one
    # Let's try to start with a greedy construction
    edges = []
    
    # Helper to check if adding edge (u, v) is valid
    # u in A, v in B
    # Need to ensure no pair (u, u') in A has 3 common neighbors including v
    # This is equivalent to: for all u' != u, |adj[u] & adj[u'] & (1 << (v-10))| <= 2
    # Since v is not in adj[u] yet, the new intersection size will be old_size + 1 if v in adj[u']
    # So we need: for all u' != u, if v in adj[u'], then |adj[u] & adj[u']| <= 1
    
    def can_add(u, v_idx):
        mask = 1 << v_idx
        for u_prime in range(10):
            if u_prime == u:
                continue
            if adj[u_prime] & mask:
                if (adj[u] & adj[u_prime]).bit_count() > 1:
                    return False
        return True

    # Greedy add edges
    # Randomize order of edges to escape local minima
    all_edges = [(u, v_idx) for u in range(10) for v_idx in range(10)]
    
    best_edges = []
    best_count = 0
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try multiple restarts
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        adj = [0] * 10
        current_edges = []
        
        for u, v_idx in all_edges:
            if can_add(u, v_idx):
                adj[u] |= (1 << v_idx)
                current_edges.append([u, v_idx + 10])
        
        # Local search: try to swap edges or add/remove
        # Actually, just try to add more edges if possible
        # The greedy above might miss some.
        # Let's do a pass to try adding any remaining edge
        improved = True
        while improved:
            improved = False
            for u in range(10):
                for v_idx in range(10):
                    if not (adj[u] & (1 << v_idx)):
                        if can_add(u, v_idx):
                            adj[u] |= (1 << v_idx)
                            current_edges.append([u, v_idx + 10])
                            improved = True
                            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
    # Convert best_edges to the required format
    # The list contains [u, v] where u in 0..9, v in 10..19
    return best_edges
