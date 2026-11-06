import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def is_regular_subset(subset, adj_masks):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        degs = []
        for v in subset:
            degs.append((adj_masks[v] & mask).bit_count())
        return len(set(degs)) == 1

    def search(n, time_limit):
        if time.time() > deadline:
            return None

        # Precompute all subsets of size 6
        subsets_6 = list(itertools.combinations(range(n), 6))
        
        # Precompute for each edge, which subsets_6 contain it
        # edge -> list of indices in subsets_6
        edge_to_subsets = {}
        for idx, subset in enumerate(subsets_6):
            for i in range(6):
                for j in range(i+1, 6):
                    u, v = subset[i], subset[j]
                    key = (min(u, v), max(u, v))
                    if key not in edge_to_subsets:
                        edge_to_subsets[key] = []
                    edge_to_subsets[key].append(idx)

        # Initialize
        adj_masks = [0] * n
        edges_set = set()
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges_set.add((u, v))

        # Initial bad_6 set
        bad_6 = set()
        for idx, subset in enumerate(subsets_6):
            if is_regular_subset(subset, adj_masks):
                bad_6.add(idx)

        while time.time() < deadline - 5:
            if not bad_6:
                # Check for larger regular subgraphs
                found_large = False
                for k in range(7, n + 1):
                    for subset in itertools.combinations(range(n), k):
                        if is_regular_subset(subset, adj_masks):
                            # Found a bad large subset
                            found_large = True
                            # Flip a random edge in this subset
                            u, v = random.sample(subset, 2)
                            key = (min(u, v), max(u, v))
                            
                            # Flip
                            if key in edges_set:
                                adj_masks[u] &= ~(1 << v)
                                adj_masks[v] &= ~(1 << u)
                                edges_set.remove(key)
                            else:
                                adj_masks[u] |= (1 << v)
                                adj_masks[v] |= (1 << u)
                                edges_set.add(key)
                            
                            # Update bad_6 for affected subsets
                            if key in edge_to_subsets:
                                for s_idx in edge_to_subsets[key]:
                                    s = subsets_6[s_idx]
                                    if is_regular_subset(s, adj_masks):
                                        bad_6.add(s_idx)
                                    else:
                                        bad_6.discard(s_idx)
                            break
                    if found_large:
                        break
                
                if not found_large:
                    # Valid graph found
                    edges = [list(e) for e in edges_set]
                    return {'n': n, 'edges': edges}
            else:
                # Pick a random bad subset from bad_6
                idx = random.choice(list(bad_6))
                subset = subsets_6[idx]
                u, v = random.sample(subset, 2)
                key = (min(u, v), max(u, v))
                
                # Flip
                if key in edges_set:
                    adj_masks[u] &= ~(1 << v)
                    adj_masks[v] &= ~(1 << u)
                    edges_set.remove(key)
                else:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges_set.add(key)
                
                # Update bad_6
                if key in edge_to_subsets:
                    for s_idx in edge_to_subsets[key]:
                        s = subsets_6[s_idx]
                        if is_regular_subset(s, adj_masks):
                            bad_6.add(s_idx)
                        else:
                            bad_6.discard(s_idx)

        return None

    graph_21 = search(21, 60)
    if graph_21:
        return graph_21

    graph_20 = search(20, 50)
    if graph_20:
        return graph_20

    return {'n': 5, 'edges': []}
