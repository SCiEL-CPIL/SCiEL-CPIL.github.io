import random
import time

def construct():
    d = 7
    n = 1 << d
    # Generate all edges
    edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
    
    # Precompute for each edge, the 6 squares it belongs to.
    # For each square, store the other 3 edges.
    edge_info = {}
    for u, v in edges:
        diff = u ^ v
        i = diff.bit_length() - 1
        others = []
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            others.append((e1, e2, e3))
        edge_info[(u, v)] = others

    def can_add(e, selected):
        """Check if edge e can be added without creating a C4."""
        for e1, e2, e3 in edge_info[e]:
            if e1 in selected and e2 in selected and e3 in selected:
                return False
        return True

    def local_search(selected, deadline):
        """Try to improve the solution by removing one edge and adding two."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try each edge in selected for removal
            edges_list = list(selected)
            random.shuffle(edges_list)
            for e_remove in edges_list:
                if time.time() >= deadline:
                    break
                # Remove e_remove
                selected.remove(e_remove)
                # Try to add edges that were previously blocked
                candidates = [e for e in edges if e not in selected and can_add(e, selected)]
                random.shuffle(candidates)
                added = 0
                for e_add in candidates:
                    if time.time() >= deadline:
                        break
                    if can_add(e_add, selected):
                        selected.add(e_add)
                        added += 1
                # If we added more than we removed, keep the improvement
                if added > 1:
                    improved = True
                else:
                    # Revert: put back e_remove and remove added edges
                    for e_add in candidates[:added]:
                        if e_add in selected:
                            selected.remove(e_add)
                    selected.add(e_remove)
        return selected

    best_edges = 0
    best_set = set()
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    while time.time() < deadline:
        selected = set()
        random.shuffle(edges)
        
        for e in edges:
            if can_add(e, selected):
                selected.add(e)
        
        # Local search polish
        remaining_time = deadline - time.time()
        if remaining_time > 5.0 and len(selected) >= best_edges - 2:
            selected = local_search(selected, time.time() + min(remaining_time * 0.3, 15.0))
        
        if len(selected) > best_edges:
            best_edges = len(selected)
            best_set = selected.copy()

    return [list(e) for e in best_set]
