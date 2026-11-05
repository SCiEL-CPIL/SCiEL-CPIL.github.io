import time
import math
import random
import sys
from collections import defaultdict

N = 128
sys.setrecursionlimit(100000)

# Precompute conflicts: for each x, conflicts[x] is a set of (u,v) pairs
# such that {x, u, v} is a forbidden triple (all three pairwise LCMs equal).
conflicts = [set() for _ in range(N + 1)]

def precompute_conflicts():
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    conflicts[a].add((b, c))
                    conflicts[b].add((a, c))
                    conflicts[c].add((a, b))

precompute_conflicts()

# Bitmask-based conflict representation
# For each x, conflict_mask[x] is a bitmask where bit i is set if i is in [1,128]
# and there exists some pair involving x and i that forms part of a forbidden triple.
# Actually, we need: for each x, which pairs (u,v) conflict with x.
# We represent this as: for each x, a list of bitmasks where each bitmask represents
# the "partner" needed. If x is added, we need no two partners both present.
# Better: conflict_pairs_mask[x] = list of (mask_u, mask_v) where mask_u has bit u, mask_v has bit v
# But that's complex. Let's use: for each x, precompute a list of (u, v) pairs.
# For fast checking: precompute for each x a set of "forbidden pairs" as bitmasks.

# Fast can_add using bitmask
# Represent current_set as a bitmask (128 bits fits in Python int)
# For each x, precompute: conflict_check[x] = list of (u_mask, v_mask) where
# u_mask = 1 << (u-1), v_mask = 1 << (v-1)
# Then can_add(x, cur_mask) checks: for any (um, vm) in conflict_check[x],
# NOT (cur_mask & um) AND NOT (cur_mask & vm) ... no wait.
# We need: NOT (both u AND v are in cur_set)
# i.e., for all (u,v) in conflicts[x], NOT ((cur_mask >> (u-1)) & 1 and (cur_mask >> (v-1)) & 1)

# Precompute for each x: a list of combined masks where bit u and bit v are both set
# Then check: (cur_mask & combined_mask) != combined_mask for all combined_masks
# Actually simpler: precompute for each x a list of (u, v) and just check membership.

# Let's precompute conflict_bitmasks[x] = list of ints where each int has exactly 2 bits set (u and v)
conflict_bitmasks = [[] for _ in range(N + 1)]
for x in range(1, N + 1):
    for u, v in conflicts[x]:
        # Use 0-indexed bits: element i -> bit (i-1)
        mask = (1 << (u - 1)) | (1 << (v - 1))
        conflict_bitmasks[x].append(mask)

def can_add_mask(x, cur_mask):
    """Check if x can be added to the set represented by cur_mask (bitmask)."""
    for cm in conflict_bitmasks[x]:
        if (cur_mask & cm) == cm:
            return False
    return True

def mask_to_set(cur_mask):
    """Convert bitmask to set of elements."""
    s = set()
    for i in range(1, N + 1):
        if (cur_mask >> (i - 1)) & 1:
            s.add(i)
    return s

def compute_conflict_graph(remaining):
    """
    Build the 2-shadow conflict graph on remaining.
    Two elements u, v in remaining are adjacent if there exists some w (any element, not necessarily in remaining)
    such that {u, v, w} is a forbidden triple.
    Actually, for the independent set formulation: we want no forbidden triple in our final set.
    If we model this as an independent set in a graph G where (u,v) is an edge if {u,v,w} is a forbidden triple for SOME w,
    that's not quite right. An independent set in G means no two elements in our set form a pair that is part of a forbidden triple with ANY third element.
    This is sufficient but not necessary. If u and v are in our set, and there is NO w in our set such that {u,v,w} is forbidden, it's fine.
    The graph edge (u,v) exists if there is ANY w in [1,128] such that {u,v,w} is forbidden.
    If we pick an independent set in this graph, then for any u,v in our set, there is no w in [1,128] such that {u,v,w} is forbidden.
    This implies there is certainly no w in our set.
    So, any independent set in this "pair-conflict" graph is a valid solution.
    This is a stronger constraint than needed (it forbids pairs that *could* form a triple with some external element),
    but it makes the problem a standard MWIS on a graph, which is much easier to bound and solve than the 3-uniform hypergraph case.
    Let's verify: if S is an independent set in G, then for any distinct u,v in S, (u,v) is not an edge in G.
    Edge (u,v) in G means: exists w in [1,128] s.t. lcm(u,v)=lcm(u,w)=lcm(v,w).
    If (u,v) is not an edge, then for all w, it is NOT the case that all three LCMs are equal.
    Therefore, for any w in S, it is not the case that {u,v,w} is a forbidden triple.
    So S contains no forbidden triple.
    This approach is valid. It might be suboptimal (smaller set), but it allows using powerful graph algorithms.
    Given N=128, the graph is sparse. Let's try this.
    """
    # We need to know for each pair (u,v) in remaining, if they are "conflicting" (i.e., part of a forbidden triple with some w).
    # We can precompute this for all pairs in [1,128].
    # pair_conflict[u][v] = True if exists w != u,v such that {u,v,w} is a forbidden triple.
    
    # Let's precompute pair_conflict globally.
    pass

# Global pair conflict matrix
# pair_conflict[i][j] = True if there exists k such that {i,j,k} is a forbidden triple.
pair_conflict = [[False] * (N + 1) for _ in range(N + 1)]

def precompute_pair_conflicts():
    # We can derive this from the triples we found in precompute_conflicts.
    # If (a,b,c) is a forbidden triple, then pair_conflict[a][b] = pair_conflict[a][c] = pair_conflict[b][c] = True.
    for a in range(1, N + 1):
        for (b, c) in conflicts[a]:
            pair_conflict[a][b] = True
            pair_conflict[a][c] = True
            pair_conflict[b][c] = True
            # Symmetry
            pair_conflict[b][a] = True
            pair_conflict[c][a] = True
            pair_conflict[c][b] = True

precompute_pair_conflicts()

def greedy_construct(deadline=None):
    """Greedy construction with multiple ordering strategies using bitmask."""
    best_mask = 0
    best_size = 0
    
    # Precompute degrees
    degrees = [0] * (N + 1)
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if pair_conflict[u][v]:
                degrees[u] += 1
                degrees[v] += 1
    
    # Strategy 1: Random orderings
    num_random = 200 if deadline is None else max(50, int(200 * min(1.0, (deadline - time.time()) / 10.0)))
    for _ in range(num_random):
        if deadline and time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        cur_mask = 0
        cur_size = 0
        for x in order:
            if can_add_mask(x, cur_mask):
                cur_mask |= (1 << (x - 1))
                cur_size += 1
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
    
    # Strategy 2: Sort by degree (low degree first)
    order_low = sorted(range(1, N + 1), key=lambda x: degrees[x])
    cur_mask = 0
    cur_size = 0
    for x in order_low:
        if can_add_mask(x, cur_mask):
            cur_mask |= (1 << (x - 1))
            cur_size += 1
    if cur_size > best_size:
        best_size = cur_size
        best_mask = cur_mask
    
    # Strategy 3: Sort by degree (high degree first)
    order_high = sorted(range(1, N + 1), key=lambda x: -degrees[x])
    cur_mask = 0
    cur_size = 0
    for x in order_high:
        if can_add_mask(x, cur_mask):
            cur_mask |= (1 << (x - 1))
            cur_size += 1
    if cur_size > best_size:
        best_size = cur_size
        best_mask = cur_mask
    
    # Strategy 4: Sort by value
    for order in [list(range(1, N + 1)), list(range(N, 0, -1))]:
        cur_mask = 0
        cur_size = 0
        for x in order:
            if can_add_mask(x, cur_mask):
                cur_mask |= (1 << (x - 1))
                cur_size += 1
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
    
    return mask_to_set(best_mask)

def local_search(s, deadline=None):
    """
    Improve set s using bitmask operations for speed.
    """
    cur_mask = 0
    for x in s:
        cur_mask |= (1 << (x - 1))
    cur_size = len(s)
    
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try to add any element
        for x in range(1, N + 1):
            if not (cur_mask >> (x - 1)) & 1:
                if can_add_mask(x, cur_mask):
                    cur_mask |= (1 << (x - 1))
                    cur_size += 1
                    improved = True
                    break
        if improved:
            continue
        
        # If no element can be added, try removing one element
        best_removal_mask = None
        best_removal_size = cur_size
        s_list = list(s)
        random.shuffle(s_list)
        
        for x in s_list:
            x_bit = 1 << (x - 1)
            new_mask = cur_mask & ~x_bit
            new_size = cur_size - 1
            # Try to greedily add elements
            for y in range(1, N + 1):
                y_bit = 1 << (y - 1)
                if not (new_mask >> (y - 1)) & 1:
                    if can_add_mask(y, new_mask):
                        new_mask |= y_bit
                        new_size += 1
            if new_size > best_removal_size:
                best_removal_size = new_size
                best_removal_mask = new_mask
                if new_size > cur_size + 1:
                    break
        
        if best_removal_mask is not None:
            cur_mask = best_removal_mask
            cur_size = best_removal_size
            improved = True
            continue
        
        # Try removing two elements
        if cur_size >= 4 and (deadline is None or time.time() < deadline - 3.0):
            best_pair_mask = None
            best_pair_size = cur_size
            s_list2 = list(s)
            random.shuffle(s_list2)
            max_pairs = 30 if deadline is None else max(5, int(30 * max(0, deadline - time.time()) / 10.0))
            pairs_tried = 0
            for i in range(len(s_list2)):
                if pairs_tried >= max_pairs:
                    break
                for j in range(i + 1, len(s_list2)):
                    if pairs_tried >= max_pairs:
                        break
                    x, y = s_list2[i], s_list2[j]
                    x_bit = 1 << (x - 1)
                    y_bit = 1 << (y - 1)
                    new_mask = cur_mask & ~x_bit & ~y_bit
                    new_size = cur_size - 2
                    for z in range(1, N + 1):
                        z_bit = 1 << (z - 1)
                        if not (new_mask >> (z - 1)) & 1:
                            if can_add_mask(z, new_mask):
                                new_mask |= z_bit
                                new_size += 1
                    pairs_tried += 1
                    if new_size > best_pair_size:
                        best_pair_size = new_size
                        best_pair_mask = new_mask
                        if new_size > cur_size:
                            break
                if best_pair_size > cur_size:
                    break
            
            if best_pair_mask is not None:
                cur_mask = best_pair_mask
                cur_size = best_pair_size
                improved = True
    
    return mask_to_set(cur_mask)

def bnb_independent_set(vertices, adjacency, deadline, best_size_ref):
    """
    Branch and Bound for Maximum Independent Set using bitmasks for speed.
    """
    best_mask = 0
    
    # Convert adjacency to bitmask representation
    adj_mask = [0] * (N + 1)
    for u in range(1, N + 1):
        for v in adjacency[u]:
            adj_mask[u] |= (1 << (v - 1))
    
    def popcount(x):
        return bin(x).count('1')
    
    def rec(cand_mask, cur_mask, cur_size):
        nonlocal best_mask, best_size_ref
        
        if time.time() > deadline:
            return
        
        # Upper bound
        if cur_size + popcount(cand_mask) <= best_size_ref[0]:
            return
        
        if cand_mask == 0:
            if cur_size > best_size_ref[0]:
                best_size_ref[0] = cur_size
                best_mask = cur_mask
            return
        
        # Pick pivot: vertex with max degree in induced subgraph
        # Iterate through candidates to find max degree
        max_deg = -1
        pivot = -1
        cm = cand_mask
        while cm:
            # Get lowest set bit
            lsb = cm & (-cm)
            v = lsb.bit_length()  # 1-indexed
            deg = popcount(cand_mask & adj_mask[v])
            if deg > max_deg:
                max_deg = deg
                pivot = v
            cm ^= lsb
        
        if pivot == -1:
            return
        
        pivot_bit = 1 << (pivot - 1)
        
        # Option 1: Include pivot
        new_cur_mask = cur_mask | pivot_bit
        new_cand_mask = cand_mask & ~pivot_bit & ~adj_mask[pivot]
        rec(new_cand_mask, new_cur_mask, cur_size + 1)
        
        # Option 2: Exclude pivot
        new_cand_mask = cand_mask & ~pivot_bit
        rec(new_cand_mask, cur_mask, cur_size)
    
    full_mask = 0
    for v in vertices:
        full_mask |= (1 << (v - 1))
    
    rec(full_mask, 0, 0)
    return mask_to_set(best_mask)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Greedy with multiple strategies + Local Search to get a good lower bound
    initial_set = greedy_construct(deadline)
    initial_set = local_search(initial_set, deadline)
    
    best_size = len(initial_set)
    best_set = initial_set
    
    # 2. Branch and Bound on the Pair-Conflict Graph
    adjacency = defaultdict(set)
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if pair_conflict[u][v]:
                adjacency[u].add(v)
                adjacency[v].add(u)
    
    vertices = list(range(1, N + 1))
    bnb_best_size = [best_size]
    
    # Run B&B with time limit
    remaining = deadline - time.time()
    if remaining > 5.0:
        bnb_result = bnb_independent_set(vertices, adjacency, deadline, bnb_best_size)
        
        if len(bnb_result) > best_size:
            best_size = len(bnb_result)
            best_set = bnb_result
    
    # 3. Final local search polish if time permits
    if time.time() < deadline - 3.0:
        best_set = local_search(best_set, deadline)
        if len(best_set) > best_size:
            best_size = len(best_set)
    
    result = list(best_set)
    if not result:
        result = [1]
        
    return result

# Test locally
if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify validity
    valid = True
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            for k in range(j + 1, len(res)):
                a, b, c = res[i], res[j], res[k]
                if math.lcm(a, b) == math.lcm(a, c) == math.lcm(b, c):
                    valid = False
                    print(f"Invalid triple: {a}, {b}, {c}")
    print(f"Valid: {valid}")
