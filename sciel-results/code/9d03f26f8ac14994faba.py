import time
import random

def construct():
    d = 7
    n = 2 ** d
    all_edges = []
    edge_dim = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                all_edges.append(e)
                edge_dim[e] = i

    # Precompute neighbors for validity check
    # For each edge e, and each j != dim(e), store the 3 other edges of the C4
    edge_neighbors = {}
    for e in all_edges:
        u, v = e
        k = edge_dim[e]
        neigh = []
        for j in range(d):
            if j == k:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            neigh.append((e1, e2, e3))
        edge_neighbors[e] = neigh

    def is_valid_add(e, S):
        for triple in edge_neighbors[e]:
            if triple[0] in S and triple[1] in S and triple[2] in S:
                return False
        return True

    def get_addable(S):
        addable = []
        for e in all_edges:
            if e not in S:
                if is_valid_add(e, S):
                    addable.append(e)
        return addable

    def get_block_counts(S):
        counts = {e: 0 for e in S}
        for e in all_edges:
            if e in S:
                continue
            # Check if e is blocked
            blocked_by = []
            for triple in edge_neighbors[e]:
                if triple[0] in S and triple[1] in S and triple[2] in S:
                    blocked_by.append(triple)
            if blocked_by:
                for triple in blocked_by:
                    counts[triple[0]] += 1
                    counts[triple[1]] += 1
                    counts[triple[2]] += 1
        return counts

    def greedy_init():
        S = set()
        random.shuffle(all_edges)
        for e in all_edges:
            if is_valid_add(e, S):
                S.add(e)
        return S

    start_time = time.time()
    deadline = start_time + 115.0
    
    best_S = greedy_init()
    best_size = len(best_S)
    
    current_S = best_S
    stagnant_count = 0
    
    while time.time() < deadline:
        addable = get_addable(current_S)
        
        if addable:
            e_add = random.choice(addable)
            current_S.add(e_add)
            stagnant_count = 0
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_S = set(current_S)
        else:
            # Stuck
            stagnant_count += 1
            
            # If stuck for too long, restart or kick
            if stagnant_count > 50:
                # Try a random kick: remove 2 random edges
                if len(current_S) > 2:
                    e_rem1, e_rem2 = random.sample(list(current_S), 2)
                    current_S.remove(e_rem1)
                    current_S.remove(e_rem2)
                    stagnant_count = 0
                else:
                    break
            else:
                counts = get_block_counts(current_S)
                if not counts:
                     # No blocked edges? Then all edges are in S?
                     # If S is all edges, it's invalid. But we maintain validity.
                     # If no edges are blocked, it means no edge in S^c is blocked?
                     # That would mean all edges in S^c are addable.
                     # But we checked addable and it was empty.
                     # This implies S^c is empty.
                     break
                
                max_count = max(counts.values())
                candidates = [e for e, c in counts.items() if c == max_count]
                if not candidates:
                    e_rem = random.choice(list(current_S))
                else:
                    e_rem = random.choice(candidates)
                
                current_S.remove(e_rem)

    return [[u, v] for u, v in best_S]
