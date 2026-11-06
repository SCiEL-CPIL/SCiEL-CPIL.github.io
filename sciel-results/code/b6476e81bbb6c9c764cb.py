import random
import time

def construct():
    d = 7
    n = 2 ** d
    total_edges = n * d // 2
    
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y, i))
                edge_to_idx[(x, y)] = idx
                edge_to_idx[(y, x)] = idx
                
    conflict_triples = [[] for _ in range(total_edges)]
    for idx, (u, v, k) in enumerate(edges):
        for i in range(d):
            if i == k:
                continue
            w = u ^ (1 << i)
            z = v ^ (1 << i)
            e1 = edge_to_idx.get((min(w, u), max(w, u)))
            e2 = edge_to_idx.get((min(z, v), max(z, v)))
            e3 = edge_to_idx.get((min(w, z), max(w, z)))
            if e1 is not None and e2 is not None and e3 is not None:
                conflict_triples[idx].append((e1, e2, e3))

    def get_conflicts(edge_idx, present_set):
        conflicts = set()
        for t in conflict_triples[edge_idx]:
            if t[0] in present_set and t[1] in present_set and t[2] in present_set:
                conflicts.update(t)
        return conflicts

    current_set = set()
    # Initial Greedy
    for idx in range(total_edges):
        if not get_conflicts(idx, current_set):
            current_set.add(idx)
            
    best_set = set(current_set)
    best_size = len(current_set)
    
    start_time = time.time()
    max_time = 110.0
    
    while time.time() - start_time < max_time:
        # Try to add a random edge
        idx = random.randint(0, total_edges - 1)
        if idx in current_set:
            continue
            
        conflicts = get_conflicts(idx, current_set)
        
        if not conflicts:
            current_set.add(idx)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                if best_size >= 308:
                    break
        else:
            # Repair: remove conflicts to make room
            # Remove 1 random conflict, check again
            removed = []
            success = False
            for _ in range(len(conflicts)):
                if not conflicts:
                    break
                r = random.choice(list(conflicts))
                current_set.remove(r)
                removed.append(r)
                conflicts.discard(r)
                
                new_conflicts = get_conflicts(idx, current_set)
                if not new_conflicts:
                    current_set.add(idx)
                    success = True
                    break
                else:
                    conflicts = new_conflicts
            
            if success:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                    if best_size >= 308:
                        break
            else:
                # Restore removed edges
                for r in removed:
                    current_set.add(r)
                    
            # Occasional Kick
            if random.random() < 0.05:
                to_remove = random.sample(list(current_set), 2)
                for r in to_remove:
                    current_set.remove(r)
                
                for _ in range(2):
                    new_idx = random.randint(0, total_edges - 1)
                    if new_idx not in current_set and not get_conflicts(new_idx, current_set):
                        current_set.add(new_idx)
                        
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)

    return [list(edges[i])[:2] for i in best_set]
