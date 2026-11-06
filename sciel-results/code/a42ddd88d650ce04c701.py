"""Combined program: fast local search to minimize colour count for K16 edge-colouring."""
import itertools
import random
import time

def construct():
    N = 16
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # edges in lexicographic order
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    E = len(edges)
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute all K4s and their edge indices
    k4s = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        k4_edges = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                    edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
        k4s.append(k4_edges)
    
    # For each edge, precompute which K4s contain it
    edge_to_k4s = [[] for _ in range(E)]
    for k4_idx, k4e in enumerate(k4s):
        for e in k4e:
            edge_to_k4s[e].append(k4_idx)
    
    def is_valid(col):
        for k4e in k4s:
            distinct = set(col[e] for e in k4e)
            if len(distinct) < 5:
                return False
        return True
    
    def palette_size(col):
        return len(set(col))
    
    # Initialize with (i+j) mod 17 – known valid with 17 colours
    colors = [(i + j) % 17 for (i, j) in edges]
    
    # Try to reduce palette via local search
    rng = random.Random(42)
    best = colors[:]
    best_score = palette_size(best)
    
    # First, try to reduce to 15, 14, 13 by remapping
    for target_max in range(15, 12, -1):
        current = colors[:]
        # Remap all colours to range [0, target_max)
        unique_cols = sorted(set(current))
        col_map = {c: i for i, c in enumerate(unique_cols)}
        for e in range(E):
            current[e] = col_map[current[e]]
        
        if not is_valid(current):
            continue
        
        cur_pal = set(current)
        if len(cur_pal) < best_score:
            best = current[:]
            best_score = len(cur_pal)
        
        # Local search to reduce further
        for iteration in range(50000):
            if time.time() > deadline:
                return best
            
            pal = sorted(set(current))
            if len(pal) <= target_max:
                if len(pal) < best_score:
                    best = current[:]
                    best_score = len(pal)
                if best_score <= 13:
                    break
                continue
            
            # Pick a random colour class to try to eliminate
            # Prefer colours with fewer edges (easier to remap)
            col_counts = {}
            for c in pal:
                col_counts[c] = sum(1 for e in range(E) if current[e] == c)
            
            # Sort by count ascending – try to eliminate small colour classes first
            sorted_cols = sorted(pal, key=lambda c: col_counts[c])
            
            found_improvement = False
            for target_color in sorted_cols:
                if time.time() > deadline:
                    break
                
                edges_with_target = [e for e in range(E) if current[e] == target_color]
                if not edges_with_target:
                    continue
                
                other_colors = [c for c in pal if c != target_color]
                if not other_colors:
                    continue
                
                # Try to remap each edge in edges_with_target to another colour
                new_current = current[:]
                success = True
                
                for e in edges_with_target:
                    rng.shuffle(other_colors)
                    found = False
                    for new_c in other_colors:
                        new_current[e] = new_c
                        # Quick check: only K4s containing edge e
                        valid_so_far = True
                        for k4_idx in edge_to_k4s[e]:
                            k4e = k4s[k4_idx]
                            distinct = set(new_current[ee] for ee in k4e)
                            if len(distinct) < 5:
                                valid_so_far = False
                                break
                        if valid_so_far:
                            found = True
                            break
                    if not found:
                        new_current[e] = current[e]
                        success = False
                        break
                
                if success:
                    # Full validation
                    if is_valid(new_current):
                        new_pal = set(new_current)
                        if len(new_pal) < len(set(current)):
                            current = new_current
                            found_improvement = True
                            if len(new_pal) < best_score:
                                best = current[:]
                                best_score = len(new_pal)
                            break
            
            if not found_improvement:
                # Try random single-edge changes to escape local optima
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    e = rng.randint(0, E - 1)
                    old_c = current[e]
                    new_c = rng.randint(0, target_max - 1)
                    if new_c == old_c:
                        continue
                    current[e] = new_c
                    # Check only K4s containing e
                    valid_so_far = True
                    for k4_idx in edge_to_k4s[e]:
                        k4e = k4s[k4_idx]
                        distinct = set(current[ee] for ee in k4e)
                        if len(distinct) < 5:
                            valid_so_far = False
                            break
                    if valid_so_far:
                        new_pal = set(current)
                        if len(new_pal) < best_score:
                            best = current[:]
                            best_score = len(new_pal)
                    else:
                        current[e] = old_c
    
    # Final safety: ensure best is valid
    if not is_valid(best):
        # Fallback to initial
        best = [(i + j) % 17 for (i, j) in edges]
    
    return best
