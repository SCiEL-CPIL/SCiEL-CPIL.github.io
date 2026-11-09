import random
import time
import itertools

N = 16
all_subsets_8 = list(itertools.combinations(range(N), 8))
splits_8 = []
for a_mask in itertools.combinations(range(8), 4):
    b_mask = [i for i in range(8) if i not in a_mask]
    splits_8.append((a_mask, b_mask))

A_verts = [u for u in range(8) if bin(u).count('1') % 2 == 0]
B_verts = [u for u in range(8) if bin(u).count('1') % 2 == 1]

q3_masks = []
for perm in itertools.permutations(range(4)):
    mask = (1 << 16) - 1
    for i in range(4):
        bit = i * 4 + perm[i]
        mask &= ~(1 << bit)
    q3_masks.append(mask)

def get_edge_count(adj, subset):
    count = 0
    for i in range(len(subset)):
        u = subset[i]
        # Count neighbors in subset
        # adj[u] is bitmask
        # Create mask for subset
        # This is slow.
        # Better:
        for j in range(i+1, len(subset)):
            v = subset[j]
            if (adj[u] >> v) & 1:
                count += 1
    return count

def has_q3(adj):
    for subset in all_subsets_8:
        if get_edge_count(adj, subset) < 12:
            continue
            
        for a_idx, b_idx in splits_8:
            a_verts = [subset[i] for i in a_idx]
            b_verts = [subset[i] for i in b_idx]
            
            mask = 0
            for i, u in enumerate(a_verts):
                for j, v in enumerate(b_verts):
                    if (adj[u] >> v) & 1:
                        mask |= (1 << (i * 4 + j))
            
            for q3_mask in q3_masks:
                if (mask & q3_mask) == q3_mask:
                    return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = -1
    best_graph = None
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        adj = [0] * N
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        while time.time() < deadline:
            if not has_q3(adj):
                break
            # Remove edge logic (simplified: just pick random edge in a Q3)
            # This part is slow.
            # We can just pick a random edge in the whole graph and remove it?
            # No, that's inefficient.
            # Let's stick to the previous logic but ensure it's not too slow.
            
            found = False
            for subset in all_subsets_8:
                if get_edge_count(adj, subset) < 12:
                    continue
                for a_idx, b_idx in splits_8:
                    a_verts = [subset[i] for i in a_idx]
                    b_verts = [subset[i] for i in b_idx]
                    mask = 0
                    edge_map = {}
                    for i, u in enumerate(a_verts):
                        for j, v in enumerate(b_verts):
                            if (adj[u] >> v) & 1:
                                bit = i * 4 + j
                                mask |= (1 << bit)
                                edge_map[bit] = (u, v)
                    
                    for q3_mask in q3_masks:
                        if (mask & q3_mask) == q3_mask:
                            q3_edges_bits = [bit for bit in range(16) if (q3_mask >> bit) & 1]
                            chosen_bit = random.choice(q3_edges_bits)
                            u, v = edge_map[chosen_bit]
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break

        for step in range(500):
            if time.time() > deadline:
                break
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u == v or ((adj[u] >> v) & 1):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3(adj):
                # Remove edge
                found = False
                for subset in all_subsets_8:
                    if get_edge_count(adj, subset) < 12:
                        continue
                    for a_idx, b_idx in splits_8:
                        a_verts = [subset[i] for i in a_idx]
                        b_verts = [subset[i] for i in b_idx]
                        mask = 0
                        edge_map = {}
                        for i, u in enumerate(a_verts):
                            for j, v in enumerate(b_verts):
                                if (adj[u] >> v) & 1:
                                    bit = i * 4 + j
                                    mask |= (1 << bit)
                                    edge_map[bit] = (u, v)
                        
                        for q3_mask in q3_masks:
                            if (mask & q3_mask) == q3_mask:
                                q3_edges_bits = [bit for bit in range(16) if (q3_mask >> bit) & 1]
                                chosen_bit = random.choice(q3_edges_bits)
                                uu, vv = edge_map[chosen_bit]
                                adj[uu] &= ~(1 << vv)
                                adj[vv] &= ~(1 << uu)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

        edge_count = sum(bin(adj[i]).count('1') for i in range(N)) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_graph = adj[:]

    if best_graph is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if (best_graph[i] >> j) & 1:
                edges.append([i, j])
    return edges
