import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    target_diffs = set(range(1, 31))
    max_elements = 32
    
    best_solution = None
    best_diameter = float('inf')
    
    # 1. Greedy Initial Solution
    def greedy_construct():
        A = [0]
        used_diffs = set()
        covered = set()
        
        for _ in range(max_elements - 1): # Add up to 31 more
            if len(covered) == 30:
                break
                
            # Find candidates
            candidates = []
            # Try to realize any missing diff
            for d in list(target_diffs - covered):
                for a in A:
                    x = a + d
                    if x in A:
                        continue
                    # Check validity
                    valid = True
                    new_diffs = []
                    for b in A:
                        diff = x - b
                        if diff <= 0:
                            continue
                        if diff in used_diffs:
                            valid = False
                            break
                        new_diffs.append(diff)
                    if valid:
                        # Count new coverage
                        nc = 0
                        for diff in new_diffs:
                            if diff in target_diffs and diff not in covered:
                                nc += 1
                        candidates.append((x, nc, new_diffs))
            
            if not candidates:
                # If no candidate covers a target, we might be stuck or need a "filler"
                # Try any valid x that minimizes x
                found = False
                for x in range(max(A) + 1, max(A) + 100):
                    valid = True
                    new_diffs = []
                    for b in A:
                        diff = x - b
                        if diff <= 0:
                            continue
                        if diff in used_diffs:
                            valid = False
                            break
                        new_diffs.append(diff)
                    if valid:
                        candidates.append((x, 0, new_diffs))
                        found = True
                        break
                if not found:
                    return None
            else:
                # Pick best candidate: most coverage, then smallest x
                candidates.sort(key=lambda c: (-c[1], c[0]))
                x, nc, nd = candidates[0]
            # Wait, the logic above is mixed. Let's simplify.
            # If candidates from targets exist, pick best.
            # If not, pick smallest valid x.
            
            if candidates:
                # Filter to those with nc > 0 if possible
                progress_candidates = [c for c in candidates if c[1] > 0]
                if progress_candidates:
                    progress_candidates.sort(key=lambda c: (-c[1], c[0]))
                    x, nc, nd = progress_candidates[0]
                else:
                    candidates.sort(key=lambda c: c[0])
                    x, nc, nd = candidates[0]
            else:
                return None

            A.append(x)
            A.sort()
            for diff in nd:
                used_diffs.add(diff)
            for diff in nd:
                if diff in target_diffs:
                    covered.add(diff)
                    
        if len(covered) == 30:
            return A
        return None

    init_A = greedy_construct()
    if init_A:
        best_solution = init_A
        best_diameter = max(init_A)
    
    # 2. Backtracking Search
    def dfs(A, used_diffs, covered):
        nonlocal best_solution, best_diameter, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        if len(covered) == 30:
            d = max(A)
            if d < best_diameter:
                best_diameter = d
                best_solution = list(A)
            return
        
        # Pruning
        if max(A) >= best_diameter:
            return
            
        if len(A) >= max_elements:
            return
            
        # Capacity Pruning
        rem_targets = len(target_diffs - covered)
        cur_size = len(A)
        slots_left = max_elements - cur_size
        # Max new diffs added by slots_left elements
        # Each new element adds cur_size + i diffs (where i is index of new element in sequence)
        # Total new diffs = slots_left * cur_size + slots_left * (slots_left - 1) / 2
        max_new_diffs = slots_left * cur_size + slots_left * (slots_left - 1) // 2
        if max_new_diffs < rem_targets:
            return

        # Generate Candidates
        # Priority: x that covers missing targets
        # Secondary: x that is small
        candidate_xs = set()
        for a in A:
            for d in target_diffs:
                if d not in covered:
                    x = a + d
                    if x > max(A):
                        candidate_xs.add(x)
        
        # Sort by x
        for x in sorted(candidate_xs):
            if x >= best_diameter:
                break
                
            # Check Sidon
            valid = True
            new_diffs = []
            for a in A:
                diff = x - a
                if diff <= 0:
                    continue
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if not valid:
                continue
            
            new_used = set(used_diffs)
            for diff in new_diffs:
                new_used.add(diff)
                
            new_covered = set(covered)
            for diff in new_diffs:
                if diff in target_diffs:
                    new_covered.add(diff)
            
            A.append(x)
            dfs(A, new_used, new_covered)
            A.pop()

    if best_solution:
        dfs([0], set(), set())
    else:
        # If greedy failed, try a broader search?
        # For now, just return empty or try a different greedy?
        # The greedy is quite robust for 1..30.
        pass

    if best_solution:
        return best_solution
    else:
        # Fallback
        return [0]

# EVOLVE-BLOCK-END
