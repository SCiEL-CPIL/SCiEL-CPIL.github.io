import time
import sys
import random
from collections import defaultdict

sys.setrecursionlimit(10000)

def construct():
    """
    Find a coloring of positions 0..127 such that every nonconstant 4-term arithmetic
    progression uses at least 3 distinct colors. Minimize the number of distinct colors.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s wall-clock limit
    
    N = 128
    
    # Precompute all 4-term arithmetic progressions
    # An AP is (a, d) -> positions a, a+d, a+2d, a+3d (0-indexed)
    aps = []
    pos_aps = [[] for _ in range(N)]
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            p1 = a
            p2 = a + d
            p3 = a + 2 * d
            p4 = a + 3 * d
            ap_idx = len(aps)
            aps.append((p1, p2, p3, p4))
            pos_aps[p1].append(ap_idx)
            pos_aps[p2].append(ap_idx)
            pos_aps[p3].append(ap_idx)
            pos_aps[p4].append(ap_idx)
    
    total_aps = len(aps)
    
    def count_distinct_colors(colors_subset):
        """Count distinct non-(-1) colors in a list."""
        s = set()
        for c in colors_subset:
            if c != -1:
                s.add(c)
        return len(s)
    
    def is_violation(aps_tuple, colors):
        """Check if a specific AP violates the constraint (uses < 3 distinct colors)."""
        c1, c2, c3, c4 = colors[aps_tuple[0]], colors[aps_tuple[1]], colors[aps_tuple[2]], colors[aps_tuple[3]]
        distinct = set()
        for c in (c1, c2, c3, c4):
            distinct.add(c)
        return len(distinct) < 3
    
    def count_violations(colors):
        """Count total number of violating APs."""
        count = 0
        for ap in aps:
            if is_violation(ap, colors):
                count += 1
        return count
    
    def try_random_walk(k, num_steps, start_colors=None, temperature=1.0):
        """
        Local search using random moves.
        k: number of colors to use (0..k-1)
        Returns best coloring found and its violation count.
        """
        if start_colors is None:
            colors = [random.randint(0, k - 1) for _ in range(N)]
        else:
            colors = list(start_colors)
        
        current_violations = count_violations(colors)
        best_colors = list(colors)
        best_violations = current_violations
        
        for step in range(num_steps):
            if time.time() > deadline:
                break
            
            # Pick a random position to change
            pos = random.randint(0, N - 1)
            old_color = colors[pos]
            
            # Pick a new color
            new_color = random.randint(0, k - 1)
            if new_color == old_color:
                continue
            
            # Apply change
            colors[pos] = new_color
            
            # Count violations for affected APs only
            # For efficiency, we could track incremental violations, but for N=128, 
            # counting all ~2000 APs is fast enough in Python if optimized.
            # Actually, let's just count all. 2000 APs * simple set op is fast.
            new_violations = count_violations(colors)
            
            # Acceptance criterion (Simulated Annealing style)
            delta = new_violations - current_violations
            if delta < 0:
                # Accept improvement
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = list(colors)
            else:
                # Accept worse with probability exp(-delta / temperature)
                # Simplified: accept with some probability to escape local minima
                if random.random() < 0.1:  # 10% chance to accept worse
                    current_violations = new_violations
                else:
                    # Revert
                    colors[pos] = old_color
        
        return best_colors, best_violations
    
    def try_hill_climbing(k, num_steps, start_colors=None):
        """
        Hill climbing: only accept improvements.
        """
        if start_colors is None:
            colors = [random.randint(0, k - 1) for _ in range(N)]
        else:
            colors = list(start_colors)
        
        current_violations = count_violations(colors)
        best_colors = list(colors)
        best_violations = current_violations
        
        for step in range(num_steps):
            if time.time() > deadline:
                break
            
            improved = False
            # Try changing each position to a better color
            # To speed up, we can iterate through positions and try all k colors
            # For N=128, k=3 or 4, this is 128*4 = 512 evaluations per step.
            # Each evaluation is ~2000 AP checks. 512 * 2000 = ~1M ops.
            # In Python, this might be slow. Let's do a smarter check.
            
            # Optimization: Only check APs affected by the change.
            for pos in range(N):
                old_color = colors[pos]
                for new_color in range(k):
                    if new_color == old_color:
                        continue
                    
                    colors[pos] = new_color
                    
                    # Check only APs involving 'pos'
                    valid = True
                    for ap_idx in pos_aps[pos]:
                        if is_violation(aps[ap_idx], colors):
                            valid = False
                            break
                    
                    if valid:
                        # This move does not introduce any NEW violations for APs involving pos.
                        # However, it might FIX violations for APs involving pos, 
                        # and it doesn't affect other APs.
                        # So the total violation count can only decrease or stay same?
                        # Wait, if we change color at pos, only APs containing pos are affected.
                        # If all those APs are now valid, the total violations decrease by 
                        # (number of APs at pos that were violating before and are now fixed).
                        # Since we don't break any OTHER APs, this is a safe improvement 
                        # if it fixes at least one violation.
                        
                        # Let's just count total violations to be safe and simple.
                        new_violations = count_violations(colors)
                        if new_violations < current_violations:
                            current_violations = new_violations
                            if new_violations < best_violations:
                                best_violations = new_violations
                                best_colors = list(colors)
                            improved = True
                            break # Move to next pos
                    
                    colors[pos] = old_color
                
                if improved:
                    # Restart the position loop to pick up new opportunities
                    break
            else:
                # No improvement found in this full pass
                # Accept random move to escape local minimum?
                # Let's just break or do a random perturbation.
                # For pure hill climbing, we stop.
                # But to allow escaping, we might randomize.
                # Let's stick to pure hill climbing for simplicity, then restart.
                break
        
        return best_colors, best_violations
    
    # Strategy:
    # 1. Try to find a valid (0 violation) coloring with k=3.
    # 2. If not found quickly, try k=4.
    # 3. Use multiple random restarts.
    
    best_overall_colors = None
    best_overall_k = 128
    
    # Try k=3 first
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        # Random start
        colors = [random.randint(0, 2) for _ in range(N)]
        
        # Phase 1: Hill climbing
        colors, viol = try_hill_climbing(3, num_steps=5000, start_colors=colors)
        
        if viol == 0:
            best_overall_colors = colors
            best_overall_k = 3
            break # Found optimal!
        
        # Phase 2: Random walk to escape local minima
        if viol < 50: # If close, try random walk
            colors, viol_rw = try_random_walk(3, num_steps=5000, start_colors=colors)
            if viol_rw == 0:
                best_overall_colors = colors
                best_overall_k = 3
                break
            if viol_rw < viol:
                colors = [c if c < 3 else random.randint(0, 2) for c in colors] # Ensure range
                # Actually try_random_walk keeps colors in 0..k-1
    
    # If k=3 failed, try k=4
    if best_overall_k > 4:
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            colors = [random.randint(0, 3) for _ in range(N)]
            colors, viol = try_hill_climbing(4, num_steps=5000, start_colors=colors)
            
            if viol == 0:
                best_overall_colors = colors
                best_overall_k = 4
                break
    
    # If k=4 failed, try k=5 (should be easy)
    if best_overall_k > 5:
        for attempt in range(5):
            if time.time() > deadline - 2:
                break
            
            colors = [random.randint(0, 4) for _ in range(N)]
            colors, viol = try_hill_climbing(5, num_steps=5000, start_colors=colors)
            
            if viol == 0:
                best_overall_colors = colors
                best_overall_k = 5
                break
    
    # Fallback: Greedy construction with many colors
    if best_overall_colors is None or best_overall_k > 5:
        # Greedy: assign color to each position that doesn't create a violation
        # with already assigned positions.
        # Since we have 128 colors available, we can always find one.
        colors = [-1] * N
        for i in range(N):
            for c in range(128):
                colors[i] = c
                valid = True
                # Check APs where i is the LAST element (p4)
                # i = a + 3d => a = i - 3d
                # p1=a, p2=a+d, p3=a+2d, p4=i
                # All previous positions a, a+d, a+2d must be assigned.
                for d in range(1, (i + 1) // 3 + 1):
                    a = i - 3 * d
                    if a >= 0:
                        p1, p2, p3, p4 = a, a + d, a + 2 * d, i
                        if colors[p1] != -1 and colors[p2] != -1 and colors[p3] != -1:
                            distinct = {colors[p1], colors[p2], colors[p3], c}
                            if len(distinct) < 3:
                                valid = False
                                break
                        # If some are unassigned, we can't check fully yet, 
                        # but since we assign in order 0..N-1, if i is p4, 
                        # then p1,p2,p3 are < i, so they are assigned.
                        # Wait, if i is p2, p3, p4 might be unassigned.
                        # But greedy forward assignment only checks constraints 
                        # that become "closed" (all assigned) or where the current 
                        # variable is the last to be assigned.
                        # For a 4-AP, the constraint is only fully checkable 
                        # when all 4 are assigned.
                        # However, we can check "immediate" violations:
                        # If 3 are assigned and 1 is current, and the 3 already 
                        # use < 2 distinct colors, then no matter what color we 
                        # pick, we might not reach 3 distinct?
                        # No, if 3 assigned use 1 distinct color, and we pick a 
                        # new color, we have 2 distinct. Still < 3.
                        # So if 3 assigned use <= 1 distinct color, it's impossible 
                        # to fix this AP by picking a color for the 4th?
                        # Wait. If p1,p2,p3 are all color 0, and p4 is color 1, 
                        # distinct = {0, 1} size 2. Violation.
                        # If p1,p2,p3 are 0,0,1, and p4 is 2, distinct = {0,1,2} size 3. OK.
                        # So we must check if it's POSSIBLE to satisfy.
                        # If assigned set has size 1, unassigned=1, max distinct=2. Fail.
                        # If assigned set has size 2, unassigned=1, max distinct=3. OK.
                        # So if len(distinct_assigned) + 1 < 3, it's impossible.
                        # i.e., if len(distinct_assigned) < 2, it's impossible.
                        
                        # Let's implement this check.
                        assigned_colors = [colors[p] for p in (p1, p2, p3) if colors[p] != -1]
                        distinct_assigned = set(assigned_colors)
                        if len(distinct_assigned) < 2:
                            valid = False
                            break
                        
                if valid:
                    break
            if colors[i] == -1:
                # Should not happen
                colors[i] = 0
        
        # Verify greedy result
        if count_violations(colors) == 0:
            best_overall_colors = colors
            best_overall_k = len(set(colors))
    
    # Remap colors to 0..k-1
    if best_overall_colors is not None:
        used = sorted(set(best_overall_colors))
        remap = {c: i for i, c in enumerate(used)}
        final_colors = [remap[c] for c in best_overall_colors]
        return final_colors
    
    # Absolute fallback (should not be reached)
    return list(range(N))

# Run construct to test locally? No, just define it.
