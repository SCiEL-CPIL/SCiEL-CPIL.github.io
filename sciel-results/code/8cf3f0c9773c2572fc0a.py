import time
import random

def construct():
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    # Helper to check if adding edge (u, v) creates a C6.
    def can_add(adj_list, u, v):
        # Quick check: degrees
        if len(adj_list[u]) < 4 or len(adj_list[v]) < 4:
            return True
        
        # DFS from u, depth 4, avoiding v and u.
        # State: (node, depth, visited_mask)
        start_mask = 1 << u
        stack = [(u, 0, start_mask)]
        
        while stack:
            node, depth, mask = stack.pop()
            
            if depth == 4:
                # Check if v is a neighbor of node.
                # Using set for O(1) lookup
                if v in adj_list[node]: # Wait, adj_list is list, not set. 
                    # We need a set for fast lookup or binary search.
                    # Let's use adj_set for lookup in can_add.
                    pass
                # Corrected: Use adj_set for lookup
                if v in adj_set[node]:
                    return False
                continue
            
            for nb in adj_list[node]:
                if nb == v:
                    continue
                if mask & (1 << nb):
                    continue
                stack.append((nb, depth + 1, mask | (1 << nb)))
        
        return True

    # Initialize with 8 disjoint K5s
    adj_list = [[] for _ in range(N)]
    adj_set = [set() for _ in range(N)]
    
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj_list[u].append(v)
                adj_list[v].append(u)
                adj_set[u].add(v)
                adj_set[v].add(u)
                
    current_edges = sum(len(s) for s in adj_set) // 2
    best_edges = current_edges
    best_adj_list = [lst[:] for lst in adj_list]
    best_adj_set = [set(s) for s in adj_set]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    missing_edges = [p for p in all_pairs if p[1] not in adj_set[p[0]]]
    
    stuck_time = start_time
    last_improvement_time = start_time
    kick_multiplier = 1.0
    
    while time.time() < deadline:
        random.shuffle(missing_edges)
        
        added_in_pass = 0
        new_missing = []
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
                
            if can_add(adj_list, u, v):
                # Add edge
                adj_list[u].append(v)
                adj_list[v].append(u)
                adj_set[u].add(v)
                adj_set[v].add(u)
                added_in_pass += 1
                current_edges += 1
            else:
                new_missing.append((u, v))
                
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj_list = [lst[:] for lst in adj_list]
            best_adj_set = [set(s) for s in adj_set]
            last_improvement_time = time.time()
            stuck_time = time.time()
            kick_multiplier = 1.0
        else:
            if added_in_pass == 0:
                stuck_time = time.time()
                kick_multiplier += 0.1
                # Cap kick multiplier
                if kick_multiplier > 3.0:
                    kick_multiplier = 3.0
                
                # Perform a kick
                if current_edges > 0:
                    kick_size = int(random.randint(2, 5) * kick_multiplier)
                    
                    existing_edges = []
                    for u in range(N):
                        for v in adj_list[u]:
                            if u < v:
                                existing_edges.append((u, v))
                            
                    random.shuffle(existing_edges)
                    
                    for _ in range(kick_size):
                        if not existing_edges:
                            break
                        u, v = existing_edges.pop()
                        adj_set[u].remove(v)
                        adj_set[v].remove(u)
                        adj_list[u].remove(v)
                        adj_list[v].remove(u)
                        current_edges -= 1
                        missing_edges.append((u, v))
            else:
                # Some edges added, but no new best. Reset stuck time?
                # No, if we didn't improve, we are still "stuck" in terms of progress.
                # But we made progress in edge count.
                # Let's reset stuck_time if we added edges.
                stuck_time = time.time()
                kick_multiplier = 1.0

        missing_edges = new_missing
        
        # Restart logic: If no improvement for 10 seconds, restart from best
        if time.time() - last_improvement_time > 10.0:
            adj_list = [lst[:] for lst in best_adj_list]
            adj_set = [set(s) for s in best_adj_set]
            current_edges = best_edges
            missing_edges = [p for p in all_pairs if p[1] not in adj_set[p[0]]]
            last_improvement_time = time.time()
            kick_multiplier = 1.0

    final_edges = []
    for u in range(N):
        for v in best_adj_list[u]:
            if u < v:
                final_edges.append([u, v])
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
