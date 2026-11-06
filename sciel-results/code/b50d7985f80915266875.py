import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    target_ns = [26, 25, 24, 23, 22, 21, 20]
    
    best_result = None
    
    for n in target_ns:
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 10:
            break
        result = local_search(n, time_limit=min(remaining - 5, 25))
        if result is not None:
            best_result = result
            break
            
    if best_result is None:
        # Fallback: Return a trivially valid small n
        # n=11 coloring from seed
        best_result = {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}
        
    return best_result

def local_search(n, time_limit=15):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge indices
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Build adjacency lists for each vertex
    adj = [[] for _ in range(n)]
    for idx, (i, j) in enumerate(edges):
        adj[i].append((j, idx))
        adj[j].append((i, idx))
    
    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Build neighbor bitmasks for each color and vertex
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
    
    def compute_total_violations(neighbors):
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total
    
    current_violations = compute_total_violations(neighbors)
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    
    # Simulated Annealing with delta-based updates
    temperature = 2.0
    cooling_rate = 0.9995
    min_temp = 0.05
    
    while time.time() < start_time + time_limit:
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        i, j = edges[e_idx]
        old_color = colors[e_idx]
        
        # Try changing to a different color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
        
        # Compute delta in violations from changing edge (i,j) from old_color to new_color
        # Remove edge from old_color:
        #   For all pairs (u,v) where both u and v are neighbors of i or j in old_color
        #   and the edge (i,j) was contributing to their common neighbors
        # 
        # When we remove edge (i,j) from old_color:
        #   - i and j are no longer neighbors in old_color
        #   - For any vertex u, if u was a common neighbor of some pair, this might change
        #
        # Actually, let's think more carefully. A violation is a pair of vertices (u,v) 
        # with >= 2 common neighbors in some color. Changing edge (i,j) from old to new:
        #
        # In old_color: removing edge (i,j) means i and j lose each other as neighbors.
        #   This can only DECREASE violations (or keep same).
        #   Specifically, for any vertex u, if u is adjacent to both i and j in old_color,
        #   then before: i and j had u as common neighbor (and possibly others).
        #   After removing (i,j): i and j are no longer connected, so they can't form a C4 together.
        #   But wait - the C4 involves 4 vertices. Let me reconsider.
        #
        # A C4 in color c is four vertices a,b,c,d such that edges (a,b),(b,c),(c,d),(d,a) all in color c.
        # Equivalently, two vertices with two common neighbors.
        #
        # When we change edge (i,j) from old_color to new_color:
        #
        # Effect on old_color:
        #   - Edge (i,j) removed. This can only remove C4s that used edge (i,j).
        #   - A C4 using edge (i,j) in old_color must have i and j as two vertices,
        #     with two common neighbors u,v such that edges (i,u),(u,j),(j,v),(v,i) exist in old_color.
        #   - Wait, that's not quite right either. Let me think about it as:
        #     i and j have common neighbors in old_color. If they have k common neighbors,
        #     then there are C(k,2) C4s involving i and j (each pair of common neighbors forms a C4).
        #     When we remove edge (i,j), i and j are no longer adjacent, so NO C4 can use both i and j.
        #     But C4s not involving both i and j are unaffected.
        #
        #   So removing edge (i,j) from old_color:
        #     - Let common_old = neighbors[old_color][i] & neighbors[old_color][j]
        #     - k = common_old.bit_count()
        #     - Number of C4s removed = C(k, 2) = k*(k-1)//2
        #     - But also, for any other pair (u,v) where u != i and v != j, the edge (i,j) 
        #       doesn't directly affect their common neighbors... unless i or j is a common neighbor.
        #     - Actually, i being a neighbor of u in old_color and j being a neighbor of u in old_color
        #       means i and j are both in neighbors[old_color][u]. But that doesn't create a C4 for (u, ?)
        #       unless there's another vertex w such that u and w share two common neighbors.
        #
        #   Hmm, this is getting complicated. Let me use a simpler delta computation:
        #
        #   Delta for removing edge (i,j) from color c:
        #     For each vertex u (u != i, u != j):
        #       - If u is adjacent to i in color c: after removal, u still adjacent to i
        #       - If u is adjacent to j in color c: after removal, u still adjacent to j
        #       - The only change: i and j are no longer adjacent to each other
        #     So the only pairs whose common neighbor count changes are pairs involving i or j.
        #     Specifically:
        #       - Pair (i, j): was adjacent, now not. But they were counted as a pair only if they had common neighbors.
        #         Actually, i and j being adjacent doesn't mean they're "common neighbors" of each other.
        #         Two vertices u,v have common neighbors if there exists w adjacent to both.
        #         i and j having common neighbors means there exists w adjacent to both i and j.
        #         When we remove edge (i,j), i and j still can have common neighbors (vertices w adjacent to both).
        #         The edge (i,j) itself doesn't affect whether i and j have common neighbors!
        #
        #   Wait, I'm confusing myself. Let me restart the delta analysis.
        #
        #   neighbors[c][v] = set of vertices u such that edge (u,v) has color c.
        #   
        #   When we change edge (i,j) from color c1 to c2:
        #     - In color c1: remove j from neighbors[c1][i] and i from neighbors[c1][j]
        #     - In color c2: add j to neighbors[c2][i] and i to neighbors[c2][j]
        #
        #   Violations count for a color c: sum over all pairs (u,v), u<v, of C(common_neighbors_c(u,v), 2)
        #   where common_neighbors_c(u,v) = neighbors[c][u] & neighbors[c][v]
        #
        #   When we modify neighbors, only pairs (u,v) where u or v is i or j can have their common neighbor set changed.
        #   Also, pairs where one of the common neighbors is i or j... no, the common neighbors are vertices,
        #   and we're changing which vertices are neighbors of i and j.
        #
        #   Let me be precise. After changing edge (i,j) from c1 to c2:
        #     - neighbors[c1][i] loses j
        #     - neighbors[c1][j] loses i  
        #     - neighbors[c2][i] gains j
        #     - neighbors[c2][j] gains i
        #
        #   For a pair (u,v) to have its common neighbor count change in color c, 
        #   either u or v must be i or j (since only neighbors[c][i] and neighbors[c][j] change),
        #   OR one of the common neighbors must be i or j (but common neighbors are determined by
        #   the intersection, and we're changing the sets, so...).
        #
        #   Actually: common_neighbors_c(u,v) = neighbors[c][u] & neighbors[c][v].
        #   If neither u nor v is i or j, then neighbors[c][u] and neighbors[c][v] are unchanged,
        #   so their intersection is unchanged. 
        #   If u = i, then common_neighbors_c(i,v) = neighbors[c][i] & neighbors[c][v].
        #   neighbors[c][i] changed (lost j in c1, gained j in c2).
        #   So for v != j: 
        #     - In c1: j was in neighbors[c1][i] but j might or might not be in neighbors[c1][v].
        #       If j is in neighbors[c1][v], then j was a common neighbor of i and v, and now it's not.
        #     - In c2: j was not in neighbors[c2][i] (before), now it is.
        #       If j is in neighbors[c2][v], then j becomes a new common neighbor.
        #
        #   This is getting complex. Let me just compute the delta by checking all affected pairs.
        #   The affected pairs are those where at least one endpoint is i or j.
        #   That's at most 2*(n-2) + 1 = 2n-3 pairs (pairs involving i or j).
        #   For each such pair, we can compute the old and new common neighbor count.
        
        # Compute delta for old_color (removing edge)
        delta_old = 0
        # Pairs involving i or j
        # Pair (i, j): 
        #   Before: common = neighbors[old_color][i] & neighbors[old_color][j]
        #   After: neighbors[old_color][i] loses j, neighbors[old_color][j] loses i
        #   But j is not in neighbors[old_color][i] in the intersection sense unless... 
        #   Wait, neighbors[old_color][i] contains j because edge (i,j) has old_color.
        #   So before: j is in neighbors[old_color][i], i is in neighbors[old_color][j].
        #   Is j in neighbors[old_color][j]? No, self-loops don't exist.
        #   Is i in neighbors[old_color][i]? No.
        #   So i is not in neighbors[old_color][i] & neighbors[old_color][j] because i not in neighbors[old_color][i].
        #   Similarly j not in the intersection.
        #   So removing i from neighbors[old_color][j] and j from neighbors[old_color][i] doesn't change
        #   the intersection neighbors[old_color][i] & neighbors[old_color][j] because i and j weren't in it.
        #   So pair (i,j) is unaffected!
        
        # For pairs (i, v) where v != j:
        #   Before: common = neighbors[old_color][i] & neighbors[old_color][v]
        #   After: neighbors[old_color][i] loses j.
        #   So if j was in neighbors[old_color][v], then j was a common neighbor, and now it's not.
        #   Change in count: -1 if j in neighbors[old_color][v], else 0.
        #   Old count: k_old, new count: k_old - (1 if j in neighbors[old_color][v] else 0)
        #   Delta in C(k,2): C(k_old - d, 2) - C(k_old, 2) where d is 0 or 1.
        
        # For pairs (j, v) where v != i:
        #   Similar: neighbors[old_color][j] loses i.
        #   If i was in neighbors[old_color][v], then i was a common neighbor, now it's not.
        
        # For pairs (v, w) where v,w != i,j: unaffected.
        
        # So delta_old = sum over v != i,j of [C(k_iv - d_iv, 2) - C(k_iv, 2)] + [C(k_jv - d_jv, 2) - C(k_jv, 2)]
        # where k_iv = (neighbors[old_color][i] & neighbors[old_color][v]).bit_count()
        #       d_iv = 1 if j in neighbors[old_color][v] else 0
        #       k_jv = (neighbors[old_color][j] & neighbors[old_color][v]).bit_count()
        #       d_jv = 1 if i in neighbors[old_color][v] else 0
        
        for v in range(n):
            if v == i or v == j:
                continue
            # Pair (i, v)
            k_iv = (neighbors[old_color][i] & neighbors[old_color][v]).bit_count()
            d_iv = 1 if (neighbors[old_color][v] >> j) & 1 else 0
            old_contrib = k_iv * (k_iv - 1) // 2 if k_iv >= 2 else 0
            new_k = k_iv - d_iv
            new_contrib = new_k * (new_k - 1) // 2 if new_k >= 2 else 0
            delta_old += new_contrib - old_contrib
            
            # Pair (j, v)
            k_jv = (neighbors[old_color][j] & neighbors[old_color][v]).bit_count()
            d_jv = 1 if (neighbors[old_color][v] >> i) & 1 else 0
            old_contrib = k_jv * (k_jv - 1) // 2 if k_jv >= 2 else 0
            new_k = k_jv - d_jv
            new_contrib = new_k * (new_k - 1) // 2 if new_k >= 2 else 0
            delta_old += new_contrib - old_contrib
        
        # Compute delta for new_color (adding edge)
        delta_new = 0
        for v in range(n):
            if v == i or v == j:
                continue
            # Pair (i, v)
            k_iv = (neighbors[new_color][i] & neighbors[new_color][v]).bit_count()
            d_iv = 1 if (neighbors[new_color][v] >> j) & 1 else 0
            old_contrib = k_iv * (k_iv - 1) // 2 if k_iv >= 2 else 0
            new_k = k_iv + d_iv
            new_contrib = new_k * (new_k - 1) // 2 if new_k >= 2 else 0
            delta_new += new_contrib - old_contrib
            
            # Pair (j, v)
            k_jv = (neighbors[new_color][j] & neighbors[new_color][v]).bit_count()
            d_jv = 1 if (neighbors[new_color][v] >> i) & 1 else 0
            old_contrib = k_jv * (k_jv - 1) // 2 if k_jv >= 2 else 0
            new_k = k_jv + d_jv
            new_contrib = new_k * (new_k - 1) // 2 if new_k >= 2 else 0
            delta_new += new_contrib - old_contrib
        
        total_delta = delta_old + delta_new
        
        # Apply the change
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)
        colors[e_idx] = new_color
        
        new_violations = current_violations + total_delta
        
        # Accept if better, or with probability exp(-delta / temperature)
        if total_delta < 0 or (temperature > min_temp and random.random() < math.exp(-total_delta / temperature)):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert
            neighbors[new_color][i] &= ~(1 << j)
            neighbors[new_color][j] &= ~(1 << i)
            neighbors[old_color][i] |= (1 << j)
            neighbors[old_color][j] |= (1 << i)
            colors[e_idx] = old_color
        
        temperature *= cooling_rate
        if temperature < min_temp:
            temperature = min_temp
            
    return None
