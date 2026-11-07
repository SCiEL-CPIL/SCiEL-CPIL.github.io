import time
import random
import sys

N = 128
DEADLINE_MARGIN = 5.0

def _build_edges():
    edges = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 > N or s2 > N or s3 > N:
                    continue
                edge = frozenset([a, b, c, s1, s2, s3])
                edges.append(edge)
    return edges

def construct():
    t_start = time.time()
    deadline = t_start + 60.0 - DEADLINE_MARGIN

    edges = _build_edges()
    num_edges = len(edges)
    
    vert_to_edges = {v: [] for v in range(1, N + 1)}
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)
            
    # Precompute edge sizes for quick access
    edge_sizes = [len(e) for e in edges]

    def _can_add(v, edge_counts):
        for ei in vert_to_edges[v]:
            # If adding v makes edge_counts[ei] equal to edge_sizes[ei], 
            # it means all vertices in the edge are now selected -> violation.
            # So we must ensure edge_counts[ei] < edge_sizes[ei] - 1.
            if edge_counts[ei] == edge_sizes[ei] - 1:
                return False
        return True

    def _update_counts(v, delta, edge_counts):
        for ei in vert_to_edges[v]:
            edge_counts[ei] += delta

    best_set = set()
    best_size = 0
    
    all_vertices = list(range(1, N + 1))
    
    # Multiple restarts to escape local minima
    # The space is small enough that we can do many restarts
    num_restarts = 1000 
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random set or heuristic
        # Heuristic: Start with upper half, then randomize
        cur_set = set(range(65, N + 1))
        
        # Randomly remove some upper half elements to allow lower half?
        # Or just start with a random subset of size ~70
        # Let's try starting with upper half and then local search
        # Sometimes it's better to start random
        if r % 2 == 0:
             cur_set = set(range(65, N + 1))
        else:
             # Random initialization biased towards upper half
             cur_set = set()
             for v in all_vertices:
                 if v > 64:
                     if random.random() < 0.9:
                         cur_set.add(v)
                 else:
                     if random.random() < 0.3:
                         cur_set.add(v)
        
        # Ensure no initial violations (should be none for upper half, 
        # but random init might have them)
        # If random init has violations, we must remove vertices to fix them
        # This is complex. Safer to start with valid set (upper half) 
        # or empty set and build up.
        
        # Let's stick to valid start: Upper half is always valid.
        # We can also start with empty and add.
        # But upper half gives 64 immediately.
        
        if r % 2 == 1:
             # Fix random init by removing violators
             # This is expensive to check. 
             # Alternative: Start empty, add greedily?
             # Or just use upper half for all restarts but perturb it.
             pass
             
        # Let's use Upper Half as base for all, then perturb.
        cur_set = set(range(65, N + 1))
        
        # Perturb: remove random elements
        for _ in range(random.randint(0, 10)):
             if cur_set:
                 v = random.choice(list(cur_set))
                 cur_set.remove(v)
                 
        # Or add random elements if possible?
        # No, start from valid state.
        
        cur_edge_counts = [0] * num_edges
        for v in cur_set:
            _update_counts(v, 1, cur_edge_counts)
            
        cur_size = len(cur_set)
        
        if cur_size > best_size:
            best_size = cur_size
            best_set = cur_set.copy()
            
        # Local Search
        stuck_count = 0
        while time.time() < deadline:
            improved = False
            
            # 1. Try to add a random missing vertex
            missing = [v for v in all_vertices if v not in cur_set]
            if missing:
                random.shuffle(missing)
                for v in missing:
                    if _can_add(v, cur_edge_counts):
                        cur_set.add(v)
                        _update_counts(v, 1, cur_edge_counts)
                        cur_size += 1
                        improved = True
                        if cur_size > best_size:
                            best_size = cur_size
                            best_set = cur_set.copy()
                        break # Add one at a time to update counts correctly? 
                        # Actually, we can add multiple if we check after each.
                        # The loop above breaks after first add. 
                        # We should loop until no more can be added.
                        
            # Re-loop to add more if we just added one
            if improved:
                stuck_count = 0
                continue
                
            # 2. If no add possible, try a swap or removal
            # Try removing a random vertex and then trying to add others
            if cur_set:
                rem_v = random.choice(list(cur_set))
                old_set = cur_set.copy()
                old_edge_counts = cur_edge_counts.copy()
                old_size = cur_size
                
                cur_set.remove(rem_v)
                _update_counts(rem_v, -1, cur_edge_counts)
                cur_size -= 1
                
                # Try to add any missing vertex
                missing_after = [v for v in all_vertices if v not in cur_set]
                random.shuffle(missing_after)
                
                local_improved = False
                for v in missing_after:
                    if _can_add(v, cur_edge_counts):
                        cur_set.add(v)
                        _update_counts(v, 1, cur_edge_counts)
                        cur_size += 1
                        local_improved = True
                        if cur_size > best_size:
                            best_size = cur_size
                            best_set = cur_set.copy()
                
                if not local_improved or cur_size <= old_size:
                    # Rollback if we didn't improve or if we just removed and couldn't add back
                    # Actually, if we removed and added back the same or fewer, it's not an improvement.
                    # But we might have added DIFFERENT vertices.
                    # If cur_size > old_size, we improved.
                    # If cur_size == old_size, we swapped. Accept swap? 
                    # Swapping doesn't increase size, but might lead to better future.
                    # Let's accept swaps with some probability (Simulated Annealing style) 
                    # or just accept if it leads to more adds later.
                    # For now, accept if size increases.
                    # If size same, we might be stuck in a local optimum of same size.
                    # Let's accept swap if it allows adding more in next step.
                    # Simplest: Accept if size >= old_size? No, that leads to random walk.
                    # Let's accept if size > old_size.
                    # If size == old_size, we can accept with probability 0.5 to escape.
                    if cur_size > old_size:
                        pass # Keep new state
                    else:
                        # Rollback
                        cur_set = old_set
                        cur_edge_counts = old_edge_counts
                        cur_size = old_size
                        stuck_count += 1
                        if stuck_count > 50:
                            break # Restart
            else:
                break

    # Final Polish: Greedy add
    if time.time() < deadline:
        be = [0] * num_edges
        for v in best_set:
            _update_counts(v, 1, be)
        
        missing_final = [v for v in all_vertices if v not in best_set]
        random.shuffle(missing_final)
        for v in missing_final:
            if _can_add(v, be):
                best_set.add(v)
                _update_counts(v, 1, be)
                
    result = [int(v) for v in sorted(best_set)]
    return result

if __name__ == "__main__":
    print("Starting...")
    res = construct()
    print(f"Size: {len(res)}")
