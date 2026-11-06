import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices
    edges = []
    edge_to_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            idx = len(edges)
            edges.append((i, j))
            edge_to_idx[(i, j)] = idx
            edge_to_idx[(j, i)] = idx
            
    # Precompute K4s
    k4s = []
    k4_edges = [] # list of lists of 6 edge indices
    k4_id = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e1 = edge_to_idx[(a, b)]
                    e2 = edge_to_idx[(a, c)]
                    e3 = edge_to_idx[(a, d)]
                    e4 = edge_to_idx[(b, c)]
                    e5 = edge_to_idx[(b, d)]
                    e6 = edge_to_idx[(c, d)]
                    k4_edges.append([e1, e2, e3, e4, e5, e6])
                    k4_id += 1
                    
    # Precompute which K4s an edge belongs to
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_k4s[e_idx].append(k4_idx)

    def check_k4(k4_idx, colors):
        e_list = k4_edges[k4_idx]
        c_set = set()
        for e in e_list:
            c_set.add(colors[e])
        return len(c_set) >= 5

    def get_violations(colors):
        viol = []
        for k4_idx in range(len(k4_edges)):
            if not check_k4(k4_idx, colors):
                viol.append(k4_idx)
        return viol

    # Deterministic greedy construction
    def greedy_construct():
        """
        Assign colors to edges in lexicographic order.
        For each edge, try colors 0, 1, 2, ... and pick the smallest
        that doesn't cause any completed K4 to have fewer than 5 distinct colors.
        A K4 is 'completed' when all 6 of its edges have been assigned.
        """
        colors = [None] * num_edges
        
        # For each edge, we need to know which K4s it's part of
        # and which other edges in those K4s are already assigned.
        
        # When assigning edge e, a K4 containing e becomes "critical" if
        # all other 5 edges of that K4 are already assigned.
        # In that case, the color of e must ensure the K4 has >= 5 distinct colors.
        
        for e_idx in range(num_edges):
            i, j = edges[e_idx]
            
            # Find all K4s that contain this edge
            affected_k4s = edge_k4s[e_idx]
            
            # For each affected K4, check if it's "almost complete"
            # i.e., all other 5 edges are already assigned
            # If so, we need to ensure adding this edge's color gives >= 5 distinct colors
            
            # Collect constraints: for each K4 where all other edges are assigned,
            # we need the color to be such that the 6 edges have >= 5 distinct colors.
            
            # A color c is invalid for edge e_idx if there exists a K4 containing e_idx
            # where all other 5 edges are assigned, and the set of those 5 colors 
            # combined with {c} has size < 5.
            
            # That means: the 5 other colors must have at most 4 distinct values,
            # AND c must be one of those values (so that adding c doesn't increase the count).
            
            # So: forbidden colors = union over all "almost complete" K4s containing e_idx
            #   of (the set of colors used on the other 5 edges, IF those 5 edges have <= 4 distinct colors)
            
            forbidden = set()
            
            for k4_idx in affected_k4s:
                e_list = k4_edges[k4_idx]
                # Check if all other edges in this K4 are assigned
                other_edges = [e for e in e_list if e != e_idx]
                if all(colors[e] is not None for e in other_edges):
                    # This K4 is almost complete
                    other_colors = [colors[e] for e in other_edges]
                    other_color_set = set(other_colors)
                    if len(other_color_set) <= 4:
                        # Any color in other_color_set would keep the distinct count <= 4
                        # We need distinct count >= 5, so we must use a color NOT in other_color_set
                        # Wait, that's not right. Let me think again.
                        
                        # The 6 edges will have colors: other_colors + [new_color]
                        # We need |set(other_colors + [new_color])| >= 5
                        # set(other_colors + [new_color]) = other_color_set ∪ {new_color}
                        # |other_color_set ∪ {new_color}| >= 5
                        # Since |other_color_set| <= 4, we need new_color ∉ other_color_set
                        # to get |other_color_set ∪ {new_color}| = |other_color_set| + 1 >= 5
                        # But if |other_color_set| = 4, then new_color ∉ other_color_set gives 5. Good.
                        # If |other_color_set| = 3, then new_color ∉ other_color_set gives 4. Not enough!
                        # If |other_color_set| = 2, then new_color ∉ other_color_set gives 3. Not enough!
                        # If |other_color_set| = 1, then new_color ∉ other_color_set gives 2. Not enough!
                        
                        # So if |other_color_set| < 4, NO color works! This means we're stuck.
                        # But wait, this shouldn't happen if we've been assigning correctly.
                        # Actually, it CAN happen. The greedy approach might paint itself into a corner.
                        
                        # Hmm, let me reconsider. If |other_color_set| = 3, we need the 6 edges to have >= 5 distinct colors.
                        # The max distinct colors we can get is 3 + 1 = 4 (if new_color is new). That's not enough.
                        # So this is a dead end. The greedy approach failed.
                        
                        # Let's return None to indicate failure.
                        return None
                        
                    else:
                        # |other_color_set| == 4 (since it's <= 4 and we handle < 4 above)
                        # Wait, I said <= 4, but I need to handle the cases properly.
                        pass
            
            # Let me redo this logic more carefully
            forbidden = set()
            dead_end = False
            
            for k4_idx in affected_k4s:
                e_list = k4_edges[k4_idx]
                other_edges = [e for e in e_list if e != e_idx]
                if all(colors[e] is not None for e in other_edges):
                    other_colors = [colors[e] for e in other_edges]
                    other_color_set = set(other_colors)
                    num_distinct = len(other_color_set)
                    
                    if num_distinct < 4:
                        # Even with a new color, max distinct is num_distinct + 1 < 5
                        dead_end = True
                        break
                    elif num_distinct == 4:
                        # We need new_color NOT in other_color_set to get 5 distinct
                        # Colors in other_color_set are forbidden
                        forbidden.update(other_color_set)
                    # If num_distinct >= 5, no constraint from this K4
        
            if dead_end:
                return None
            
            # Try colors starting from 0
            for c in range(100):  # Upper bound on colors to try
                if c not in forbidden:
                    colors[e_idx] = c
                    break
            else:
                # Couldn't find a valid color
                return None
        
        # Compress colors
        used_colors = sorted(set(colors))
        mapping = {c: i for i, c in enumerate(used_colors)}
        compressed = [mapping[c] for c in colors]
        
        # Verify
        for k4_idx in range(len(k4_edges)):
            e_list = k4_edges[k4_idx]
            c_set = set(colors[e] for e in e_list)
            if len(c_set) < 5:
                return None  # Should not happen
        
        return compressed

    # Try greedy first
    if time.time() < deadline - 60:
        res = greedy_construct()
        if res is not None:
            return res

    # Fallback to random search
    def try_k_set(K, time_limit):
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        violations = set(get_violations(colors))
        
        while time.time() < time_limit and violations:
            v_idx = random.choice(list(violations))
            e_list = k4_edges[v_idx]
            
            c_vals = [colors[e] for e in e_list]
            c_set = set(c_vals)
            
            freq = {}
            for c in c_vals:
                freq[c] = freq.get(c, 0) + 1
            
            dup_color = None
            for c, f in freq.items():
                if f >= 2:
                    dup_color = c
                    break
                    
            if dup_color is None:
                continue

            candidate_edges = [e for e in e_list if colors[e] == dup_color]
            e_to_change = random.choice(candidate_edges)
            
            available = [c for c in range(K) if c not in c_set]
            if available:
                new_color = random.choice(available)
            else:
                new_color = random.randint(0, K - 1)
                
            colors[e_to_change] = new_color
            
            # Update violations
            affected = edge_k4s[e_to_change]
            
            # Check v_idx
            if check_k4(v_idx, colors):
                violations.discard(v_idx)
            else:
                violations.add(v_idx) # Already there
            
            # Check other affected
            for k4_idx in affected:
                if k4_idx == v_idx:
                    continue
                if check_k4(k4_idx, colors):
                    violations.discard(k4_idx)
                else:
                    violations.add(k4_idx)
                    
        if not violations:
            used_colors = sorted(list(set(colors)))
            mapping = {c: i for i, c in enumerate(used_colors)}
            compressed = [mapping[c] for c in colors]
            return compressed
        return None

    # Try K=13
    if time.time() < deadline - 30:
        res = try_k_set(13, time.time() + 30)
        if res:
            return res

    # Try K=14
    if time.time() < deadline - 20:
        res = try_k_set(14, time.time() + 20)
        if res:
            return res

    # Try K=15
    if time.time() < deadline - 10:
        res = try_k_set(15, time.time() + 10)
        if res:
            return res

    # Fallback: Baseline
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
