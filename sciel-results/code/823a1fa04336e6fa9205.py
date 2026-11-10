"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # 90s wall-clock, leave 5s margin
    
    N = 256
    max_sum = 2 * N
    perfect_squares = set()
    for k in range(1, int(math.isqrt(max_sum)) + 2):
        sq = k * k
        if sq <= max_sum:
            perfect_squares.add(sq)
    
    # Self-loop check: 2a is a perfect square
    # 2a = k^2 => a = k^2/2, so k must be even, k=2m, a=2m^2
    self_loop_set = set()
    for m in range(1, 12):
        a = 2 * m * m
        if 1 <= a <= N:
            self_loop_set.add(a)
    
    # Remaining vertices (no self-loops)
    remaining = [a for a in range(1, N + 1) if a not in self_loop_set]
    n_rem = len(remaining)
    
    # The conflict graph is bipartite: odd numbers connect to even numbers (and vice versa)
    # because odd + even = odd, and perfect squares that are odd are 1, 9, 25, 49, 81, 121, 169, 225
    # Wait - let me check: odd + even can be odd or even depending on which is larger
    # Actually: odd + even = odd. Even + odd = odd. 
    # Perfect squares can be odd or even.
    # But the key insight: if a and b have the same parity, a+b is even.
    # If a and b have different parity, a+b is odd.
    # 
    # For the graph to be bipartite, we need that edges only go between two partitions.
    # Let's check: can two odd numbers sum to a perfect square? Yes: 1+3=4, 1+7=8(no), 1+15=16
    # Can two even numbers sum to a perfect square? Yes: 2+6=8(no), 2+14=16
    # 
    # So the graph is NOT necessarily bipartite by parity!
    # 
    # Let me reconsider. The directive says to model as max flow with bipartite graph.
    # Let me think about this differently.
    #
    # Actually, wait. Let me re-examine. The problem is finding maximum independent set.
    # For general graphs, this is NP-hard. But if the graph is bipartite, we can use
    # Kőnig's theorem: max independent set = n - min vertex cover = n - max matching.
    #
    # Let me verify if our graph is bipartite by trying to 2-color it.
    # If we can 2-color it, it's bipartite.
    
    # Build adjacency list
    adj = [[] for _ in range(n_rem)]
    for i in range(n_rem):
        a = remaining[i]
        for j in range(i + 1, n_rem):
            b = remaining[j]
            if (a + b) in perfect_squares:
                adj[i].append(j)
                adj[j].append(i)
    
    # Try to 2-color the graph to verify bipartiteness
    color = [-1] * n_rem
    is_bipartite = True
    for start in range(n_rem):
        if color[start] == -1:
            color[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break
    
    if is_bipartite:
        # Use Kőnig's theorem via maximum bipartite matching
        # Partition vertices by color
        left_vertices = [i for i in range(n_rem) if color[i] == 0]
        right_vertices = [i for i in range(n_rem) if color[i] == 1]
        
        # Map right vertices to indices 0..len(right)-1
        right_idx = {v: i for i, v in enumerate(right_vertices)}
        n_left = len(left_vertices)
        n_right = len(right_vertices)
        
        # Build matching: for each left vertex, which right vertices are adjacent
        left_to_right = [[] for _ in range(n_left)]
        for i in range(n_rem):
            if color[i] == 0:
                li = left_vertices.index(i)  # This is O(n) each time, let's optimize
                for j in adj[i]:
                    if color[j] == 1:
                        left_to_right[li].append(right_idx[j])
        
        # Actually, let's rebuild this more efficiently
        left_idx_map = {v: i for i, v in enumerate(left_vertices)}
        left_to_right = [[] for _ in range(n_left)]
        for i in range(n_rem):
            if color[i] == 0:
                li = left_idx_map[i]
                for j in adj[i]:
                    if color[j] == 1:
                        left_to_right[li].append(right_idx[j])
        
        # Hopcroft-Karp maximum bipartite matching
        def hopcroft_karp():
            pair_u = [-1] * n_left  # pair_u[u] = matched right vertex
            pair_v = [-1] * n_right  # pair_v[v] = matched left vertex
            dist = [0] * n_left
            
            def bfs():
                queue = deque()
                for u in range(n_left):
                    if pair_u[u] == -1:
                        dist[u] = 0
                        queue.append(u)
                    else:
                        dist[u] = float('inf')
                found = False
                while queue:
                    u = queue.popleft()
                    for v in left_to_right[u]:
                        if pair_v[v] == -1:
                            found = True
                        elif dist[pair_v[v]] == float('inf'):
                            dist[pair_v[v]] = dist[u] + 1
                            queue.append(pair_v[v])
                return found
            
            def dfs(u):
                for v in left_to_right[u]:
                    if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                        pair_u[u] = v
                        pair_v[v] = u
                        return True
                dist[u] = float('inf')
                return False
            
            matching = 0
            while bfs():
                for u in range(n_left):
                    if pair_u[u] == -1:
                        if dfs(u):
                            matching += 1
            
            return matching, pair_u, pair_v
        
        matching, pair_u, pair_v = hopcroft_karp()
        
        # By Kőnig's theorem: max independent set size = n_rem - matching
        # To find the actual independent set:
        # Minimum vertex cover: 
        #   U = left vertices not matched (or matched but not reachable in alternating forest)
        #   Actually, standard construction:
        #   Let Z be set of vertices reachable from unmatched left vertices via alternating paths
        #   Min vertex cover = (Left \ Z) ∪ (Right ∩ Z)
        #   Max independent set = V \ (Min vertex cover) = (Left ∩ Z) ∪ (Right \ Z)
        
        # Find alternating forest from unmatched left vertices
        in_z = [False] * n_left  # in_z[u] = u is in Z (left side)
        in_z_right = [False] * n_right  # in_z_right[v] = v is in Z (right side)
        
        queue = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                in_z[u] = True
                queue.append(('L', u))
        
        while queue:
            side, node = queue.popleft()
            if side == 'L':
                u = node
                for v in left_to_right[u]:
                    if not in_z_right[v]:
                        in_z_right[v] = True
                        if pair_v[v] != -1:
                            u2 = pair_v[v]
                            if not in_z[u2]:
                                in_z[u2] = True
                                queue.append(('L', u2))
            else:
                v = node
                # From right vertex v, go to its matched left vertex
                if pair_v[v] != -1:
                    u = pair_v[v]
                    if not in_z[u]:
                        in_z[u] = True
                        queue.append(('L', u))
        
        # Max independent set = (Left ∩ Z) ∪ (Right \ Z)
        independent_set = []
        for u in range(n_left):
            if in_z[u]:
                independent_set.append(left_vertices[u])
        for v in range(n_right):
            if not in_z_right[v]:
                independent_set.append(right_vertices[v])
        
        # Convert to actual values
        result = sorted([remaining[i] for i in independent_set])
        
        # Validate
        valid = True
        for i in range(len(result)):
            a = result[i]
            if (2 * a) in perfect_squares:
                valid = False
                break
            for j in range(i + 1, len(result)):
                b = result[j]
                if (a + b) in perfect_squares:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return result
    
    # Fallback: if graph is not bipartite or validation fails, use heuristic approach
    # Build adjacency as bitmasks for speed
    adj_mask = [0] * n_rem
    for i in range(n_rem):
        a = remaining[i]
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    def greedy_min_degree(rs=None):
        """Greedy: repeatedly pick vertex with smallest degree among remaining"""
        if rs is None:
            rs = random.Random()
        
        available = (1 << n_rem) - 1
        result_indices = []
        
        while available:
            min_deg = n_rem + 1
            candidates = []
            avail_list = []
            temp = available
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                avail_list.append(i)
                temp ^= lsb
            
            for i in avail_list:
                deg = bin(adj_mask[i] & available).count('1')
                if deg < min_deg:
                    min_deg = deg
                    candidates = [i]
                elif deg == min_deg:
                    candidates.append(i)
            
            v = rs.choice(candidates)
            result_indices.append(v)
            available &= ~(1 << v)
            available &= ~adj_mask[v]
        
        return result_indices
    
    def local_improvement_fast(indices, rs, max_iters=2000):
        """Local improvement using bitmask operations for speed"""
        if not indices:
            return indices
        
        selected_mask = 0
        for i in indices:
            selected_mask |= (1 << i)
        
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            non_selected = ((1 << n_rem) - 1) & ~selected_mask
            non_sel_list = []
            temp = non_selected
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                non_sel_list.append(i)
                temp ^= lsb
            
            rs.shuffle(non_sel_list)
            
            for v in non_sel_list:
                conflicts = adj_mask[v] & selected_mask
                if conflicts == 0:
                    selected_mask |= (1 << v)
                    improved = True
                else:
                    conflict_count = bin(conflicts).count('1')
                    net_gain = 1 - conflict_count
                    if net_gain > 0:
                        selected_mask = (selected_mask & ~conflicts) | (1 << v)
                        improved = True
        
        result = []
        temp = selected_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            result.append(i)
            temp ^= lsb
        
        return result
    
    best_indices = greedy_min_degree(random.Random(0))
    best_indices = local_improvement_fast(best_indices, random.Random(1), max_iters=2000)
    
    for seed in range(50):
        if time.time() > DEADLINE:
            break
        s = greedy_min_degree(random.Random(seed + 100))
        s = local_improvement_fast(s, random.Random(seed + 200), max_iters=1000)
        if len(s) > len(best_indices):
            best_indices = s
    
    result = sorted([remaining[i] for i in best_indices])
    
    # Validate
    valid = True
    for i in range(len(result)):
        a = result[i]
        if (2 * a) in perfect_squares:
            valid = False
            break
        for j in range(i + 1, len(result)):
            b = result[j]
            if (a + b) in perfect_squares:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return [1]
    
    return result

# EVOLVE-BLOCK-END
