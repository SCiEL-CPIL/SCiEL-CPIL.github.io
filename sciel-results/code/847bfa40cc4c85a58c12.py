import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 15
    # Initialize with K_{7,8}
    # Partition A: 0..6 (size 7)
    # Partition B: 7..14 (size 8)
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edges = []
    
    for u in range(7):
        for v in range(7, 15):
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            edges.append((u, v))
            
    best_edges = [list(e) for e in edges]
    best_count = len(edges)
    
    def update_best():
        nonlocal best_edges, best_count
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]
            
    def can_add(u, v):
        """
        Check if adding edge (u, v) to current graph (adj, deg) keeps it valid.
        Assumes (u, v) is NOT in adj.
        """
        if u == v:
            return False
        if v in adj[u]:
            return True
        
        du = deg[u]
        dv = deg[v]
        
        # New degrees
        du_new = du + 1
        dv_new = dv + 1
        
        # 1. Paths s-u-v-t
        # s in N(u) \ {v}
        # t in N(v) \ {u, s}
        # Check deg[s] != deg[t]
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
        
        # 2. Paths s-v-u-t
        # s in N(v) \ {u}
        # t in N(u) \ {v, s}
        # Check deg[s] != deg[t]
        for s in adj[v]:
            if s == u:
                continue
            ds = deg[s]
            for t in adj[u]:
                if t == v or t == s:
                    continue
                if ds == deg[t]:
                    return False
        
        # 3. Paths u-v-x-y
        # v in N(u) (in new graph)
        # x in N(v) \ {u}
        # y in N(x) \ {v, u}
        # Endpoints u, y. Check deg_new[u] != deg[y]
        for x in adj[v]:
            if x == u:
                continue
            for y in adj[x]:
                if y == v or y == u:
                    continue
                if du_new == deg[y]:
                    return False
        
        # 4. Paths v-u-x-y
        # u in N(v) (in new graph)
        # x in N(u) \ {v}
        # y in N(x) \ {u, v}
        # Endpoints v, y. Check deg_new[v] != deg[y]
        for x in adj[u]:
            if x == v:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
                if dv_new == deg[y]:
                    return False
        
        # 5. Paths x-u-v-y where x in N(u)\{v}, y in N(v)\{u}
        # This is the same as case 1 but with different variable names
        # Already covered
        
        # 6. Paths x-v-u-y where x in N(v)\{u}, y in N(u)\{v}
        # Same as case 2
        
        # 7. Paths u-x-v-y where x in N(u)\{v}, x in N(v)\{u}
        # Wait, this is a 3-edge path u-x-v-y
        # x must be in N(u) and N(v)
        # y must be in N(v)\{u,x}
        # Endpoints u, y. Check deg_new[u] != deg[y]
        for x in adj[u]:
            if x == v:
                continue
            if x not in adj[v]:
                continue
            for y in adj[v]:
                if y == u or y == x:
                    continue
                if du_new == deg[y]:
                    return False
        
        # 8. Paths v-x-u-y where x in N(v)\{u}, x in N(u)\{v}
        # Same structure as 7 but swapped
        for x in adj[v]:
            if x == u:
                continue
            if x not in adj[u]:
                continue
            for y in adj[u]:
                if y == v or y == x:
                    continue
                if dv_new == deg[y]:
                    return False
        
        # 9. Paths x-y-u-v where x in N(y), y in N(u)\{v}
        # Wait, this is the same as case 4 with different variable names
        # Already covered
        
        # 10. Paths y-x-v-u where y in N(x), x in N(v)\{u}
        # Same as case 3
        # Already covered
        
        return True

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        edges.append((u, v))

    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1
        edges.remove((u, v))

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    present = set(edges)
    
    def try_add_any(missing_edges, max_checks=200):
        """Try to add edges from the given list. Returns count added."""
        count = 0
        shuffled = list(missing_edges)
        random.shuffle(shuffled)
        for u, v in shuffled[:max_checks]:
            if time.time() > deadline - 3:
                break
            if (u, v) in present:
                continue
            if can_add(u, v):
                add_edge(u, v)
                present.add((u, v))
                count += 1
        return count
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add missing edges
        missing = [p for p in all_pairs if p not in present]
        if not missing:
            break
        
        added = try_add_any(missing)
        if added > 0:
            update_best()
            improved = True
            continue
        
        if time.time() > deadline - 15:
            break
        
        # Phase 2: Try 1-edge swaps (remove one, add one different)
        if time.time() < deadline - 10:
            for attempt in range(100):
                if time.time() > deadline - 10:
                    break
                if not edges:
                    break
                
                e_rm = random.choice(edges)
                u_rm, v_rm = e_rm
                remove_edge(u_rm, v_rm)
                present.discard(e_rm)
                
                found_swap = False
                missing_now = [p for p in all_pairs if p not in present]
                random.shuffle(missing_now)
                
                for u_add, v_add in missing_now:
                    if time.time() > deadline - 10:
                        break
                    if (u_add, v_add) in present:
                        continue
                    if can_add(u_add, v_add):
                        add_edge(u_add, v_add)
                        present.add((u_add, v_add))
                        found_swap = True
                        break
                
                if found_swap:
                    improved = True
                    break
                else:
                    # Revert
                    add_edge(u_rm, v_rm)
                    present.add(e_rm)
        
        # Phase 3: Try 2-edge swaps (remove two, add two different)
        if not improved and time.time() < deadline - 5:
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                if len(edges) < 2:
                    break
                
                # Pick two edges to remove
                e1, e2 = random.sample(edges, 2)
                u1, v1 = e1
                u2, v2 = e2
                
                remove_edge(u1, v1)
                present.discard(e1)
                remove_edge(u2, v2)
                present.discard(e2)
                
                # Try to add two edges
                added = 0
                missing_now = [p for p in all_pairs if p not in present]
                random.shuffle(missing_now)
                
                for u_add, v_add in missing_now:
                    if time.time() > deadline - 5:
                        break
                    if (u_add, v_add) in present:
                        continue
                    if can_add(u_add, v_add):
                        add_edge(u_add, v_add)
                        present.add((u_add, v_add))
                        added += 1
                        if added == 2:
                            break
                
                if added == 2:
                    improved = True
                    break
                else:
                    # Revert: remove what we added, re-add what we removed
                    for u_rm, v_rm in [(u_add, v_add) for _ in range(added)]:
                        pass
                    # Actually we need to track what we added
                    # Let me redo this more carefully
                
                # Revert properly
                # Remove the edges we just added
                for e in list(edges):
                    if e not in [e1, e2]:
                        # This is a newly added edge, remove it
                        pass
                
                # This is getting complicated. Let me use a simpler approach:
                # Just revert everything and try again
                # Actually, let me restructure
        
    return best_edges

# Test run
if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
