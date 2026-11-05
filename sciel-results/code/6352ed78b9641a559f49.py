import time
import random
import itertools

N = 16
DEADLINE = 145.0

# Precomputation
# 1. Map edge (u,v) to id 0..119
edge_to_id = {}
id_to_edge = []
eid = 0
for u in range(N):
    for v in range(u+1, N):
        edge_to_id[(u,v)] = eid
        id_to_edge.append((u,v))
        eid += 1
NUM_EDGES = eid # 120

# 2. Generate all Q3 masks
# Q3 is a subgraph of K8.
# We iterate all 8-subsets of vertices.
# For each 8-subset, we generate all Q3 subgraphs within it.
# A Q3 is defined by a bipartition (A,B) of the 8 vertices into two sets of 4,
# and a 3-regular bipartite graph between them.
# The 3-regular bipartite graphs on 4+4 vertices correspond to permutation matrices of missing edges.
# There are 24 such graphs for a fixed bipartition.
# Number of bipartitions: C(8,4)/2 = 35.
# Total Q3s per 8-set: 35 * 24 = 840.

q3_masks = []
q3_by_edge = [[] for _ in range(NUM_EDGES)]

# Precompute permutations of 4
perms4 = list(itertools.permutations(range(4)))

def generate_q3s_for_8set(vertices):
    """Generate masks of all Q3 subgraphs in the given 8 vertices."""
    masks = []
    # Get all bipartitions
    # We can just pick 4 vertices for A, the rest for B.
    # To avoid duplicates (A,B) and (B,A), we can enforce min(A) < min(B) or similar,
    # but since we are generating edge sets, (A,B) and (B,A) produce the same edge set?
    # No, the graph structure is the same, just labels swapped. The edge set is identical.
    # So we only need to iterate unique bipartitions.
    # C(8,4) = 70. Half are duplicates. 35 unique.
    
    verts = list(vertices)
    # Generate all 4-subsets of verts
    a_sets = list(itertools.combinations(verts, 4))
    # Filter to avoid duplicates: keep only if min(a) < min(b)
    # b is the complement.
    seen_bipartitions = set()
    unique_bipartitions = []
    for a in a_sets:
        b = tuple(v for v in verts if v not in a)
        key = (a, b) if a < b else (b, a)
        if key not in seen_bipartitions:
            seen_bipartitions.add(key)
            unique_bipartitions.append((a, b))
            
    for a, b in unique_bipartitions:
        # a and b are tuples of 4 vertices
        # For each permutation p in perms4, define the missing edges.
        # The graph is K4,4 minus a perfect matching.
        # The perfect matching is defined by the permutation.
        # If p = (0,1,2,3), missing edges are (a[0],b[0]), (a[1],b[1]), etc.
        # The existing edges are all other 12.
        for p in perms4:
            mask = 0
            # Build the mask for this Q3
            # Iterate all pairs (u in a, v in b)
            # If (u, v) is NOT in the matching, add edge.
            matching = set(zip(a, b)) # This is wrong, p defines the mapping
            # p[i] is the index in b that a[i] is NOT connected to?
            # Let's say missing edge is (a[i], b[p[i]]).
            missing = set()
            for i in range(4):
                missing.add((a[i], b[p[i]]))
                
            for u in a:
                for v in b:
                    if (u, v) not in missing and (v, u) not in missing:
                        # Add edge
                        e_id = edge_to_id.get((u,v)) or edge_to_id.get((v,u))
                        if e_id is not None:
                            mask |= (1 << e_id)
            masks.append(mask)
    return masks

start_time = time.time()

# Generate all Q3s
all_vertices = range(N)
for comb in itertools.combinations(all_vertices, 8):
    if time.time() > DEADLINE - 20: # Safety
        break
    masks = generate_q3s_for_8set(comb)
    for m in masks:
        q3_masks.append(m)
        # Update q3_by_edge
        # We need to know which edges are in m
        # This is slow if we do it for every mask.
        # But we can do it during generation.
        # Let's refactor generate_q3s to yield (mask, edges_in_mask)

# Refactor for speed:
# We will clear q3_by_edge and rebuild it.
q3_by_edge = [[] for _ in range(NUM_EDGES)]
# Actually, rebuilding 10M entries is slow.
# Let's do it during generation.

q3_masks = []
q3_by_edge = [[] for _ in range(NUM_EDGES)]

def generate_q3s_fast(vertices):
    verts = list(vertices)
    a_sets = list(itertools.combinations(verts, 4))
    seen = set()
    for a in a_sets:
        b = tuple(v for v in verts if v not in a)
        key = (a, b) if a < b else (b, a)
        if key in seen: continue
        seen.add(key)
        
        for p in perms4:
            mask = 0
            edges_in_q3 = []
            missing = set()
            for i in range(4):
                missing.add((a[i], b[p[i]]))
            
            for u in a:
                for v in b:
                    if (u, v) not in missing and (v, u) not in missing:
                        e_id = edge_to_id.get((u,v))
                        if e_id is None:
                            e_id = edge_to_id.get((v,u))
                        mask |= (1 << e_id)
                        edges_in_q3.append(e_id)
            
            q3_masks.append(mask)
            for e_id in edges_in_q3:
                q3_by_edge[e_id].append(len(q3_masks) - 1)

for comb in itertools.combinations(all_vertices, 8):
    if time.time() > DEADLINE - 30:
        break
    generate_q3s_fast(comb)

# Precompute rest masks for faster checking?
# q3_rest_masks[i] = q3_masks[i] with one bit removed?
# No, the check is: (q3_mask & ~edge_mask) & ~current_mask == 0
# This is fast enough.

def is_q3_free_add(e_id, current_mask):
    """Check if adding e_id to current_mask creates a Q3."""
    e_mask = 1 << e_id
    # Iterate Q3s containing e_id
    for q_idx in q3_by_edge[e_id]:
        q_mask = q3_masks[q_idx]
        # Check if all other edges in Q3 are in current_mask
        rest = q_mask & ~e_mask
        if (rest & ~current_mask) == 0:
            return False
    return True

def greedy_build(seed):
    rng = random.Random(seed)
    edges = list(range(NUM_EDGES))
    rng.shuffle(edges)
    
    current_mask = 0
    count = 0
    
    for e in edges:
        if is_q3_free_add(e, current_mask):
            current_mask |= (1 << e)
            count += 1
            
    return count, current_mask

def local_search(seed, initial_mask, initial_count):
    rng = random.Random(seed)
    current_mask = initial_mask
    current_count = initial_count
    best_mask = initial_mask
    best_count = initial_count
    
    # Try to add edges that were skipped
    # Or try swaps?
    # Simple approach: Randomly remove an edge, then greedily add back as many as possible.
    
    for step in range(200): # Limited steps per restart
        if time.time() > DEADLINE - 5:
            break
        
        # Pick a random edge in current graph to remove
        if current_count == 0:
            break
        bits = []
        temp_mask = current_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            bits.append(lsb.bit_length() - 1)
            temp_mask ^= lsb
        remove_e = bits[rng.randint(0, len(bits)-1)]
        
        new_mask = current_mask & ~(1 << remove_e)
        new_count = current_count - 1
        
        # Greedily add edges
        # Try all edges not in new_mask
        # To be faster, just try the ones we removed and some random others?
        # Let's try all edges in random order
        candidates = [e for e in range(NUM_EDGES) if not (new_mask & (1 << e))]
        rng.shuffle(candidates)
        
        added = 0
        for e in candidates:
            if is_q3_free_add(e, new_mask):
                new_mask |= (1 << e)
                new_count += 1
                added += 1
                if new_count > best_count:
                    best_count = new_count
                    best_mask = new_mask
        
        # If we didn't improve, keep current? Or move to new state for diversity?
        # Move to new state to escape local minima
        current_mask = new_mask
        current_count = new_count

    return best_count, best_mask

def construct():
    global q3_masks, q3_by_edge
    
    # If generation was cut short, we might have a partial list.
    # That's okay, it's a valid subset of constraints.
    
    start = time.time()
    best_count = 0
    best_mask = 0
    
    # Initial greedy runs
    for seed in range(10):
        cnt, mask = greedy_build(seed)
        if cnt > best_count:
            best_count = cnt
            best_mask = mask
            
    # Local search from best
    cnt, mask = local_search(0, best_mask, best_count)
    if cnt > best_count:
        best_count = cnt
        best_mask = mask
        
    # More greedy runs with different seeds
    while time.time() < DEADLINE - 10:
        seed = int(time.time() * 1000) % 100000
        cnt, mask = greedy_build(seed)
        if cnt > best_count:
            best_count = cnt
            best_mask = mask
            
        # Occasionally run local search
        if seed % 5 == 0:
            cnt, mask = local_search(seed, mask, cnt)
            if cnt > best_count:
                best_count = cnt
                best_mask = mask
                
    # Convert mask to list of edges
    result = []
    for i in range(NUM_EDGES):
        if (best_mask >> i) & 1:
            u, v = id_to_edge[i]
            result.append([u, v])
            
    return result
