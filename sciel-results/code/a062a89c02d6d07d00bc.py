import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Fast 2-colorability check. Returns True if colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True

    m = len(edges)
    edge_verts = [tuple(e) for e in edges]

    # v_to_e_mask[v] = bitmask of edges containing v
    v_to_e_mask = [0] * n
    for i, edge in enumerate(edges):
        for v in edge:
            v_to_e_mask[v] |= (1 << i)

    def solve():
        colors = [-1] * n
        e0 = [0] * m
        e1 = [0] * m

        def backtrack():
            # Heuristic: Pick the uncolored vertex involved in the most "tight" edges
            # An edge is tight if it has 3 or more colored vertices.
            best_v = -1
            best_score = -1
            for v in range(n):
                if colors[v] != -1:
                    continue
                score = 0
                mm = v_to_e_mask[v]
                while mm:
                    lb = mm & (-mm)
                    ei = lb.bit_length() - 1
                    if e0[ei] + e1[ei] >= 3:
                        score += 1
                    mm ^= lb
                if score > best_score:
                    best_score = score
                    best_v = v
                if best_score >= 10: # Early break for speed
                    break

            if best_v == -1:
                return True

            v = best_v

            # Try colors 0 and 1
            # Heuristic: Try the color that appears more often in neighboring edges?
            # Or just 0 then 1.
            
            for try_color in [0, 1]:
                undo_log = []
                colors[v] = try_color
                undo_log.append(('v', v, -1))

                conflict = False
                queue = []
                mm = v_to_e_mask[v]
                while mm:
                    lb = mm & (-mm)
                    ei = lb.bit_length() - 1
                    queue.append(ei)
                    mm ^= lb

                visited_edges = set(queue)
                qi = 0
                while qi < len(queue):
                    ei = queue[qi]
                    qi += 1
                    verts = edge_verts[ei]
                    c0 = 0
                    c1 = 0
                    uncolored = []
                    for uv in verts:
                        cv = colors[uv]
                        if cv == 0:
                            c0 += 1
                        elif cv == 1:
                            c1 += 1
                        else:
                            uncolored.append(uv)

                    old_e0 = e0[ei]
                    old_e1 = e1[ei]
                    e0[ei] = c0
                    e1[ei] = c1
                    undo_log.append(('e0', ei, old_e0))
                    undo_log.append(('e1', ei, old_e1))

                    if c0 == 5 or c1 == 5:
                        conflict = True
                        break

                    # Unit Propagation
                    if c0 == 4:
                        for uv in uncolored:
                            if colors[uv] == -1:
                                colors[uv] = 1
                                undo_log.append(('v', uv, -1))
                                mm2 = v_to_e_mask[uv]
                                while mm2:
                                    lb2 = mm2 & (-mm2)
                                    ej = lb2.bit_length() - 1
                                    if ej not in visited_edges:
                                        queue.append(ej)
                                        visited_edges.add(ej)
                                    mm2 ^= lb2
                            elif colors[uv] == 0:
                                conflict = True
                                break
                        if conflict:
                            break
                    elif c1 == 4:
                        for uv in uncolored:
                            if colors[uv] == -1:
                                colors[uv] = 0
                                undo_log.append(('v', uv, -1))
                                mm2 = v_to_e_mask[uv]
                                while mm2:
                                    lb2 = mm2 & (-mm2)
                                    ej = lb2.bit_length() - 1
                                    if ej not in visited_edges:
                                        queue.append(ej)
                                        visited_edges.add(ej)
                                    mm2 ^= lb2
                            elif colors[uv] == 1:
                                conflict = True
                                break
                        if conflict:
                            break

                if not conflict:
                    if backtrack():
                        return True

                for item in reversed(undo_log):
                    if item[0] == 'v':
                        colors[item[1]] = item[2]
                    elif item[0] == 'e0':
                        e0[item[1]] = item[2]
                    elif item[0] == 'e1':
                        e1[item[1]] = item[2]

            return False

        return backtrack()

    return solve()

def _get_witness_coloring(n, edges):
    """
    Returns a valid 2-coloring if exists, else None.
    Reuses logic from _is_2_colorable but returns the colors.
    """
    if not edges:
        return [0] * n

    m = len(edges)
    edge_verts = [tuple(e) for e in edges]
    v_to_e_mask = [0] * n
    for i, edge in enumerate(edges):
        for v in edge:
            v_to_e_mask[v] |= (1 << i)

    def solve():
        colors = [-1] * n
        e0 = [0] * m
        e1 = [0] * m

        def backtrack():
            best_v = -1
            best_score = -1
            for v in range(n):
                if colors[v] != -1:
                    continue
                score = 0
                mm = v_to_e_mask[v]
                while mm:
                    lb = mm & (-mm)
                    ei = lb.bit_length() - 1
                    if e0[ei] + e1[ei] >= 3:
                        score += 1
                    mm ^= lb
                if score > best_score:
                    best_score = score
                    best_v = v
                if best_score >= 10:
                    break

            if best_v == -1:
                return True

            v = best_v

            for try_color in [0, 1]:
                undo_log = []
                colors[v] = try_color
                undo_log.append(('v', v, -1))

                conflict = False
                queue = []
                mm = v_to_e_mask[v]
                while mm:
                    lb = mm & (-mm)
                    ei = lb.bit_length() - 1
                    queue.append(ei)
                    mm ^= lb

                visited_edges = set(queue)
                qi = 0
                while qi < len(queue):
                    ei = queue[qi]
                    qi += 1
                    verts = edge_verts[ei]
                    c0 = 0
                    c1 = 0
                    uncolored = []
                    for uv in verts:
                        cv = colors[uv]
                        if cv == 0:
                            c0 += 1
                        elif cv == 1:
                            c1 += 1
                        else:
                            uncolored.append(uv)

                    old_e0 = e0[ei]
                    old_e1 = e1[ei]
                    e0[ei] = c0
                    e1[ei] = c1
                    undo_log.append(('e0', ei, old_e0))
                    undo_log.append(('e1', ei, old_e1))

                    if c0 == 5 or c1 == 5:
                        conflict = True
                        break

                    if c0 == 4:
                        for uv in uncolored:
                            if colors[uv] == -1:
                                colors[uv] = 1
                                undo_log.append(('v', uv, -1))
                                mm2 = v_to_e_mask[uv]
                                while mm2:
                                    lb2 = mm2 & (-mm2)
                                    ej = lb2.bit_length() - 1
                                    if ej not in visited_edges:
                                        queue.append(ej)
                                        visited_edges.add(ej)
                                    mm2 ^= lb2
                            elif colors[uv] == 0:
                                conflict = True
                                break
                        if conflict:
                            break
                    elif c1 == 4:
                        for uv in uncolored:
                            if colors[uv] == -1:
                                colors[uv] = 0
                                undo_log.append(('v', uv, -1))
                                mm2 = v_to_e_mask[uv]
                                while mm2:
                                    lb2 = mm2 & (-mm2)
                                    ej = lb2.bit_length() - 1
                                    if ej not in visited_edges:
                                        queue.append(ej)
                                        visited_edges.add(ej)
                                    mm2 ^= lb2
                            elif colors[uv] == 1:
                                conflict = True
                                break
                        if conflict:
                            break

                if not conflict:
                    if backtrack():
                        return True

                for item in reversed(undo_log):
                    if item[0] == 'v':
                        colors[item[1]] = item[2]
                    elif item[0] == 'e0':
                        e0[item[1]] = item[2]
                    elif item[0] == 'e1':
                        e1[item[1]] = item[2]

            return False

        if backtrack():
            # Fill any remaining uncolored vertices (should be none if logic is correct, but just in case)
            for i in range(n):
                if colors[i] == -1:
                    colors[i] = 0
            return colors
        return None

    return solve()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

    best_edges_count = float('inf')
    best_result = None

    # Fallback: Complete graph on 9 vertices (126 edges)
    edges_9 = [list(e) for e in combinations(range(9), 5)]
    best_result = {'n': 9, 'edges': edges_9}
    best_edges_count = len(edges_9)

    # Strategy: "Add to break, then Prune"
    # 1. Start with a small random set of edges.
    # 2. Check if 2-colorable.
    # 3. If yes, get witness coloring. Find monochromatic edges in the full set.
    # 4. Add one such edge.
    # 5. If no (non-2-colorable), try to prune edges.
    
    for n in [9, 10, 11, 12, 13]:
        if time.time() > deadline:
            break

        all_possible = list(combinations(range(n), 5))
        all_possible_set = set(all_possible)

        for trial in range(20):
            if time.time() > deadline:
                break

            # Initialize with a random subset of edges
            # Size varies: smaller for larger n?
            # Let's start with ~20-30 edges
            initial_size = 20
            current_edges = set(random.sample(all_possible, initial_size))
            
            # Main loop: Iterate until non-2-colorable, then prune
            # We limit the number of "Add" steps to avoid getting stuck with too many edges
            max_add_steps = 50
            add_steps = 0
            
            while time.time() < deadline:
                # Check colorability
                if _is_2_colorable(n, list(current_edges)):
                    # It is colorable. Get witness.
                    coloring = _get_witness_coloring(n, list(current_edges))
                    if coloring is None:
                        break # Should not happen
                    
                    # Find monochromatic edges not in current set
                    candidates = []
                    for e in all_possible:
                        if e in current_edges:
                            continue
                        # Check if monochromatic under coloring
                        c0 = 0
                        c1 = 0
                        mono = True
                        for v in e:
                            if coloring[v] == 0:
                                c0 += 1
                            else:
                                c1 += 1
                            if c0 > 0 and c1 > 0:
                                mono = False
                                break
                        if mono:
                            candidates.append(e)
                    
                    if not candidates:
                        # No monochromatic edges available? 
                        # This implies the current set is "maximal" w.r.t this coloring?
                        # Actually, if no new edge is mono, we can't break this coloring by adding one edge.
                        # We need to add a set of edges? Or this coloring is robust.
                        # Let's just add a random edge to force change.
                        remaining = [e for e in all_possible if e not in current_edges]
                        if not remaining:
                            break
                        current_edges.add(random.choice(remaining))
                    else:
                        # Pick a random candidate, or the one with highest degree?
                        # Random is fine for diversity.
                        current_edges.add(random.choice(candidates))
                    
                    add_steps += 1
                    if add_steps > max_add_steps:
                        break
                else:
                    # It is NON-2-colorable.
                    # Try to prune
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        # Random order for pruning
                        edges_list = list(current_edges)
                        random.shuffle(edges_list)
                        for e in edges_list:
                            if time.time() > deadline:
                                break
                            # Try removing e
                            temp_edges = current_edges - {e}
                            if not _is_2_colorable(n, list(temp_edges)):
                                current_edges = temp_edges
                                improved = True
                                # Continue pruning immediately
                                break # Restart loop with new set
                            
                    # Update best
                    if len(current_edges) < best_edges_count:
                        best_edges_count = len(current_edges)
                        best_result = {'n': n, 'edges': [list(e) for e in current_edges]}
                        if best_edges_count <= 30:
                            break
                    break # Move to next trial

            if best_edges_count <= 30:
                break

    if best_result is None:
        return {'n': 9, 'edges': edges_9}

    return best_result
