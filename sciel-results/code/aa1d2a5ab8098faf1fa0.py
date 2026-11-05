import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles as tuples of 4 edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = (edge_index(a, b), edge_index(b, c), edge_index(c, d), edge_index(d, a))
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c-a
                    e2 = (edge_index(a, b), edge_index(b, d), edge_index(d, c), edge_index(c, a))
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d-a
                    e3 = (edge_index(a, c), edge_index(c, b), edge_index(b, d), edge_index(d, a))
                    cycles.append(e3)
    return cycles

_CYCLES = build_four_cycles()
NUM_CYCLES = len(_CYCLES)
_CYCLES_BY_EDGE = [[] for _ in range(NUM_EDGES)]
for idx, cyc in enumerate(_CYCLES):
    for e in cyc:
        _CYCLES_BY_EDGE[e].append(idx)

def is_valid_cycle(colors, cyc):
    """Check if a 4-cycle has at most 3 distinct colors."""
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    if c0 == c1 or c0 == c2 or c0 == c3:
        return True
    if c1 == c2 or c1 == c3:
        return True
    if c2 == c3:
        return True
    return False

def validate(colors):
    """Check that no 4-cycle has 4 distinct colors. Returns True if valid."""
    for cyc in _CYCLES:
        if not is_valid_cycle(colors, cyc):
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def csp_backtracking(deadline, rng, max_depth=120):
    """
    CSP backtracking with forward checking and most-constrained-first ordering.
    Tries to assign colors to edges one by one, preferring unused colors.
    Uses forward checking to detect conflicts early.
    """
    colors = [-1] * NUM_EDGES
    used_colors = set()
    
    # Precompute for each edge, which other edges share a cycle with it
    # and what constraints that imposes
    
    def get_unassigned_edges():
        return [e for e in range(NUM_EDGES) if colors[e] == -1]
    
    def constraint_tightness(e):
        """
        For an unassigned edge e, compute how constrained it is.
        For each cycle containing e, count how many other edges in the cycle
        are already assigned. More assigned neighbors = tighter constraint.
        """
        tightness = 0
        for cyc_idx in _CYCLES_BY_EDGE[e]:
            cyc = _CYCLES[cyc_idx]
            assigned_count = 0
            for other_e in cyc:
                if other_e != e and colors[other_e] != -1:
                    assigned_count += 1
            tightness += assigned_count
        return tightness
    
    def try_assign(e, c):
        """Try to assign color c to edge e, checking all cycles containing e."""
        colors[e] = c
        valid = True
        for cyc_idx in _CYCLES_BY_EDGE[e]:
            cyc = _CYCLES[cyc_idx]
            # Check if all edges in cycle are assigned
            all_assigned = True
            for other_e in cyc:
                if colors[other_e] == -1:
                    all_assigned = False
                    break
            if all_assigned:
                if not is_valid_cycle(colors, cyc):
                    valid = False
                    break
        if not valid:
            colors[e] = -1
            return False
        return True
    
    def forward_check(e, c):
        """
        After assigning color c to edge e, check if any unassigned edge
        still has at least one valid color option.
        Returns True if forward checking passes.
        """
        # For each unassigned edge that shares a cycle with e,
        # check if it still has at least one possible color
        for other_e in range(NUM_EDGES):
            if colors[other_e] != -1:
                continue
            # Check if other_e shares any cycle with e
            shares_cycle = False
            for cyc_idx in _CYCLES_BY_EDGE[e]:
                if other_e in _CYCLES[cyc_idx]:
                    shares_cycle = True
                    break
            if not shares_cycle:
                continue
            
            # Check if other_e has at least one valid color
            has_valid = False
            for test_c in range(MAX_COLOR + 1):
                # Quick check: does this color create an immediate conflict?
                colors[other_e] = test_c
                ok = True
                for cyc_idx in _CYCLES_BY_EDGE[other_e]:
                    cyc = _CYCLES[cyc_idx]
                    all_assigned = True
                    for ee in cyc:
                        if colors[ee] == -1:
                            all_assigned = False
                            break
                    if all_assigned:
                        if not is_valid_cycle(colors, cyc):
                            ok = False
                            break
                colors[other_e] = -1
                if ok:
                    has_valid = True
                    break
            if not has_valid:
                return False
        return True
    
    def backtrack(depth):
        """Recursive backtracking search."""
        if time.time() > deadline:
            return False
        
        if depth >= NUM_EDGES:
            return True
        
        # Get unassigned edges and order by constraint tightness (most constrained first)
        unassigned = get_unassigned_edges()
        if not unassigned:
            return True
        
        # Sort by tightness (descending), then by index for determinism
        unassigned.sort(key=lambda e: (-constraint_tightness(e), e))
        
        # Pick the most constrained edge
        e = unassigned[0]
        
        # Generate candidate colors: prefer unused colors
        unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
        rng.shuffle(unused)
        
        used_list = list(used_colors)
        rng.shuffle(used_list)
        
        # Try unused colors first, then used colors
        candidates = unused[:30] + used_list[:20]
        
        for c in candidates:
            if time.time() > deadline:
                return False
            
            if try_assign(e, c):
                was_unused = c not in used_colors
                if was_unused:
                    used_colors.add(c)
                
                if forward_check(e, c):
                    if backtrack(depth + 1):
                        return True
                
                if was_unused:
                    # Only remove if this edge was the only one using this color
                    still_used = any(colors[i] == c for i in range(NUM_EDGES) if i != e)
                    if not still_used:
                        used_colors.discard(c)
                
                colors[e] = -1
        
        return False
    
    # Run backtracking
    success = backtrack(0)
    
    if success:
        return colors
    else:
        # If backtracking failed (timeout or no solution), return what we have
        # Fill in any unassigned edges with color 0
        for e in range(NUM_EDGES):
            if colors[e] == -1:
                colors[e] = 0
        return colors

def local_improve_aggressive(colors, rng, deadline):
    """
    Aggressive local search to maximize distinct colors.
    """
    colors = colors[:]
    used_colors = set(colors)
    current_count = len(used_colors)
    best_count = current_count
    best_colors = colors[:]
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Strategy 1: Try to introduce new colors by recoloring single edges
        for _ in range(300):
            if time.time() >= deadline - 2:
                break
            
            e = rng.randint(0, NUM_EDGES - 1)
            old = colors[e]
            
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if not unused:
                break
            
            rng.shuffle(unused)
            for new_c in unused[:30]:
                if new_c == old:
                    continue
                colors[e] = new_c
                valid = True
                for cyc_idx in _CYCLES_BY_EDGE[e]:
                    cyc = _CYCLES[cyc_idx]
                    if not is_valid_cycle(colors, cyc):
                        valid = False
                        break
                if valid:
                    used_colors.add(new_c)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = colors[:]
                    improved = True
                    break
                else:
                    colors[e] = old
        
        # Strategy 2: Try 2-edge moves
        for _ in range(150):
            if time.time() >= deadline - 2:
                break
            
            e1 = rng.randint(0, NUM_EDGES - 1)
            shared_edges = set()
            for cyc_idx in _CYCLES_BY_EDGE[e1]:
                cyc = _CYCLES[cyc_idx]
                for other_e in cyc:
                    if other_e != e1:
                        shared_edges.add(other_e)
            
            if not shared_edges:
                continue
            
            e2 = rng.choice(list(shared_edges))
            old1, old2 = colors[e1], colors[e2]
            
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if not unused:
                break
            
            rng.shuffle(unused)
            for new_c1 in unused[:15]:
                colors[e1] = new_c1
                valid1 = True
                for cyc_idx in _CYCLES_BY_EDGE[e1]:
                    cyc = _CYCLES[cyc_idx]
                    if not is_valid_cycle(colors, cyc):
                        valid1 = False
                        break
                if not valid1:
                    colors[e1] = old1
                    continue
                
                # Try to also recolor e2
                unused2 = [c for c in range(MAX_COLOR + 1) if c not in used_colors and c != new_c1]
                if unused2:
                    rng.shuffle(unused2)
                    for new_c2 in unused2[:10]:
                        colors[e2] = new_c2
                        valid2 = True
                        for cyc_idx in _CYCLES_BY_EDGE[e2]:
                            cyc = _CYCLES[cyc_idx]
                            if not is_valid_cycle(colors, cyc):
                                valid2 = False
                                break
                        if valid2:
                            used_colors.add(new_c1)
                            used_colors.add(new_c2)
                            current_count += 2
                            if current_count > best_count:
                                best_count = current_count
                                best_colors = colors[:]
                            improved = True
                            break
                        else:
                            colors[e2] = old2
                    else:
                        # Only e1 gets new color
                        used_colors.add(new_c1)
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_colors = colors[:]
                        improved = True
                        break
                else:
                    used_colors.add(new_c1)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = colors[:]
                    improved = True
                    break
                break
            else:
                colors[e1] = old1
        
        # Rebuild used_colors for consistency
        used_colors = set(colors)
        current_count = len(used_colors)
    
    return best_colors, best_count

def construct():
    start = time.time()
    deadline = start + 55  # Safety margin before 60s limit
    
    rng = random.Random(42)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # First, try CSP backtracking with a generous time budget
    csp_colors = csp_backtracking(deadline - 30, rng)
    if validate(csp_colors):
        csp_count = count_colors(csp_colors)
        if csp_count > best_count:
            best_count = csp_count
            best_colors = csp_colors[:]
    
    # Then apply local improvement to the CSP result
    if time.time() < deadline - 5:
        improved_c, improved_cnt = local_improve_aggressive(best_colors, rng, deadline)
        if improved_cnt > best_count:
            best_count = improved_cnt
            best_colors = improved_c[:]
    
    # If CSP didn't give us a good result, try multiple random greedy constructions
    if best_count < 15 and time.time() < deadline - 10:
        edge_indices = list(range(NUM_EDGES))
        
        for trial in range(10):
            if time.time() >= deadline - 5:
                break
            
            shuffled = list(edge_indices)
            rng.shuffle(shuffled)
            
            colors = [-1] * NUM_EDGES
            used_colors = set()
            
            for e in shuffled:
                unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
                rng.shuffle(unused)
                used_list = list(used_colors)
                rng.shuffle(used_list)
                all_candidates = unused[:20] + used_list[:10]
                
                found = False
                for c in all_candidates:
                    colors[e] = c
                    valid = True
                    for cyc_idx in _CYCLES_BY_EDGE[e]:
                        cyc = _CYCLES[cyc_idx]
                        if colors[cyc[0]] == -1 or colors[cyc[1]] == -1 or colors[cyc[2]] == -1 or colors[cyc[3]] == -1:
                            continue
                        if not is_valid_cycle(colors, cyc):
                            valid = False
                            break
                    if valid:
                        found = True
                        break
                
                if not found:
                    colors[e] = 0
                else:
                    used_colors.add(colors[e])
            
            if validate(colors):
                cnt = count_colors(colors)
                if cnt > best_count:
                    best_count = cnt
                    best_colors = colors[:]
    
    # Final validation
    if not validate(best_colors):
        best_colors = [0] * NUM_EDGES
        best_count = 1
    
    return best_colors
