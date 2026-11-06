import time
import numpy as np
import random

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # 1. Generate all divisors of 51975 > 1
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for a in range(4):
        for b in range(3):
            for c in range(2):
                for d in range(2):
                    v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                    if v > 1:
                        divisors.append(v)
    divisors.sort()
    
    # Map divisor to index for quick lookup
    div_index = {m: i for i, m in enumerate(divisors)}
    n_divs = len(divisors)

    # 2. Precompute positions for each class (m, r)
    # pos[m_idx][r] is an array of indices in [0, base) where idx % m == r
    positions = []
    for m in divisors:
        arr = []
        for r in range(m):
            # np.arange(r, base, m) gives all x in [0, base) such that x % m == r
            arr.append(np.arange(r, base, m, dtype=np.int32))
        positions.append(arr)

    # 3. Helper functions
    
    def calculate_uncovered(covered):
        return int((covered == 0).sum())

    def greedy_init(deadline):
        """
        Greedy set cover initialization.
        Returns (classes, covered, coverage_count, used_moduli)
        """
        covered = np.zeros(base, dtype=np.uint8)
        coverage_count = np.zeros(base, dtype=np.uint16) # Max classes is < 256
        used_moduli = set()
        classes = []
        
        # We want to pick the class with max gain.
        # To speed this up, we can pre-sort divisors by size (smaller m covers more).
        # But dynamic gain depends on current coverage.
        # We scan all available classes each iteration.
        # N_classes = sum(m for m in divisors) = 12480.
        # Iterations <= n_divs = 48.
        # 48 * 12480 = ~600k operations. Fast enough with NumPy.
        
        while time.time() < deadline:
            best_gain = 0
            best_m_idx = -1
            best_r = -1
            
            # Iterate through all available moduli
            for m_idx, m in enumerate(divisors):
                if m in used_moduli:
                    continue
                
                # Iterate through residues for this modulus
                # Optimization: If the modulus is very large, it covers few points.
                # We can skip if max possible gain is <= best_gain.
                # Max gain for (m, r) is base // m.
                if base // m <= best_gain:
                    continue
                    
                for r in range(m):
                    pos = positions[m_idx][r]
                    # Gain is number of uncovered points in pos
                    gain = int((covered[pos] == 0).sum())
                    if gain > best_gain:
                        best_gain = gain
                        best_m_idx = m_idx
                        best_r = r
            
            if best_gain == 0:
                break
            
            m = divisors[best_m_idx]
            pos = positions[best_m_idx][best_r]
            
            classes.append([m, best_r])
            used_moduli.add(m)
            covered[pos] = 1
            coverage_count[pos] += 1
            
        return classes, covered, coverage_count, used_moduli

    def local_search(classes, covered, coverage_count, used_moduli, deadline):
        """
        Local search to improve the solution.
        Moves: Add, Remove, Swap.
        """
        best_classes = [c[:] for c in classes]
        best_uncovered = calculate_uncovered(covered)
        
        # We maintain state for the current best
        cur_classes = [c[:] for c in classes]
        cur_covered = covered.copy()
        cur_count = coverage_count.copy()
        cur_used = set(used_moduli)
        cur_uncovered = best_uncovered
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # --- Phase 1: Try to Remove redundant classes ---
            # A class is redundant if removing it does not change the uncovered set.
            # i.e., all positions it covers are covered by at least one other class.
            # Check: min(coverage_count[pos]) >= 2 for all pos in class?
            # If so, we can remove it.
            
            removed_any = False
            for i in range(len(cur_classes)):
                if time.time() > deadline:
                    break
                m, r = cur_classes[i]
                m_idx = div_index[m]
                pos = positions[m_idx][r]
                
                # If all positions in this class are covered by >= 2 classes, it's redundant
                # We need to check if any position has count == 1.
                # If min count is 1, removing it will uncover those positions.
                # If min count >= 2, removing it is safe (uncovered count stays same).
                # Actually, we want to minimize uncovered. Removing a redundant class 
                # keeps uncovered count same but reduces number of classes (not scored, 
                # but might help future swaps? No, score is only uncovered count).
                # However, the problem asks to minimize uncovered residues.
                # Removing redundant classes doesn't improve score.
                # So we should only remove if it *improves* score? No, removal never improves score.
                # Wait, the score is "number of residues ... covered by no class".
                # So minimizing uncovered is the goal.
                # Removing a class can only increase or keep same uncovered.
                # So Phase 1 (Removal) is only useful if we want to free up a modulus for a Swap.
                # But Swap handles that.
                # So we can skip pure removals unless they are part of a swap logic.
                # Let's focus on Add and Swap.
                pass
            
            # --- Phase 2: Try to Add a new class ---
            best_add_gain = 0
            best_add_m = -1
            best_add_r = -1
            best_add_pos = None
            
            for m_idx, m in enumerate(divisors):
                if m in cur_used:
                    continue
                # Heuristic: skip if max possible gain is small
                if base // m <= best_add_gain:
                    continue
                    
                for r in range(m):
                    pos = positions[m_idx][r]
                    gain = int((cur_covered[pos] == 0).sum())
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add_m = m
                        best_add_r = r
                        best_add_pos = pos
            
            if best_add_gain > 0:
                # Perform Add
                cur_classes.append([best_add_m, best_add_r])
                cur_used.add(best_add_m)
                cur_covered[best_add_pos] = 1
                cur_count[best_add_pos] += 1
                cur_uncovered -= best_add_gain
                
                if cur_uncovered < best_uncovered:
                    best_uncovered = cur_uncovered
                    best_classes = [c[:] for c in cur_classes]
                improved = True
                continue # Restart loop to check for more adds/removals
            
            # --- Phase 3: Try to Swap ---
            # Replace (m_old, r_old) with (m_new, r_new)
            # Gain = new_uncovered - old_uncovered < 0
            # new_uncovered = cur_uncovered - gain_new + loss_old
            # We want gain_new > loss_old
            
            # Precompute loss for each existing class
            # loss_old = number of positions in (m_old, r_old) that are covered ONLY by this class.
            # i.e., count of pos in pos_old where cur_count[pos] == 1.
            
            losses = []
            for i, (m, r) in enumerate(cur_classes):
                m_idx = div_index[m]
                pos = positions[m_idx][r]
                loss = int((cur_count[pos] == 1).sum())
                losses.append((loss, i, m, r, pos))
            
            # Sort by loss ascending (try removing low loss ones first)
            losses.sort(key=lambda x: x[0])
            
            # Precompute gains for all unused classes
            gains = []
            for m_idx, m in enumerate(divisors):
                if m in cur_used:
                    continue
                if base // m <= 0:
                    continue
                for r in range(m):
                    pos = positions[m_idx][r]
                    gain = int((cur_covered[pos] == 0).sum())
                    if gain > 0:
                        gains.append((gain, m, r, pos))
            
            # Sort by gain descending
            gains.sort(key=lambda x: x[0], reverse=True)
            
            # Try top swaps
            # Limit iterations to avoid timeout
            found_swap = False
            # Only check top 50 gains and top 50 losses to keep it fast
            for g_val, m_new, r_new, pos_new in gains[:50]:
                if g_val <= 0:
                    break
                for l_val, i_old, m_old, r_old, pos_old in losses[:50]:
                    if g_val > l_val:
                        # Perform Swap
                        # 1. Remove old
                        cur_classes.pop(i_old)
                        cur_used.discard(m_old)
                        
                        # Update coverage
                        # Decrease count for pos_old
                        cur_count[pos_old] -= 1
                        # Positions that become uncovered are those where count becomes 0
                        newly_uncovered = (cur_count[pos_old] == 0).sum()
                        
                        # 2. Add new
                        cur_classes.append([m_new, r_new])
                        cur_used.add(m_new)
                        
                        # Update coverage
                        # Increase count for pos_new
                        cur_count[pos_new] += 1
                        # Positions that become covered are those where count becomes 1
                        newly_covered = (cur_count[pos_new] == 1).sum()
                        
                        # Net change in uncovered
                        # old_uncovered -> new_uncovered
                        # new_uncovered = cur_uncovered - newly_covered + newly_uncovered
                        # Note: newly_covered is subset of pos_new that were 0 before.
                        # newly_uncovered is subset of pos_old that were 1 before.
                        
                        # Wait, cur_uncovered is the count of 0s in cur_covered.
                        # We need to update cur_covered as well?
                        # It's easier to just recalculate cur_uncovered from cur_covered if we modify it.
                        # But modifying cur_covered is O(N) if we do it naively.
                        # Let's update cur_covered directly.
                        
                        # Reset cur_covered to match cur_count? No, cur_covered is binary.
                        # We can update cur_covered based on changes.
                        
                        # For pos_old: if count becomes 0, set covered to 0.
                        idxs_old_uncovered = pos_old[cur_count[pos_old] == 0]
                        cur_covered[idxs_old_uncovered] = 0
                        
                        # For pos_new: if count becomes 1, set covered to 1.
                        idxs_new_covered = pos_new[cur_count[pos_new] == 1]
                        cur_covered[idxs_new_covered] = 1
                        
                        # Recalculate uncovered count efficiently?
                        # We know:
                        # lost_covered = newly_uncovered (positions in old that became 0)
                        # gained_covered = newly_covered (positions in new that became 1)
                        # But there might be overlap?
                        # pos_old and pos_new are from different moduli.
                        # They can overlap.
                        
                        # It is safer to just recalculate if the change is small,
                        # or maintain a variable.
                        # Let's just recalculate cur_uncovered using the delta.
                        # cur_uncovered += lost_covered - gained_covered
                        # lost_covered is count of pos_old where count went 1->0.
                        # gained_covered is count of pos_new where count went 0->1.
                        
                        # However, if pos_old and pos_new overlap, a position might be in both.
                        # If p in pos_old and pos_new:
                        # Old: count >= 1.
                        # New: count (after remove) >= 0, then (after add) >= 1.
                        # If count was 1 in old, and p is in new:
                        # Remove: count 1->0. (Uncovered increases by 1)
                        # Add: count 0->1. (Uncovered decreases by 1)
                        # Net 0.
                        
                        # My delta calculation:
                        # newly_uncovered: count of p in pos_old where final count is 0.
                        # newly_covered: count of p in pos_new where final count is 1.
                        
                        # This logic is tricky with overlaps.
                        # Given the small size of pos arrays (max base/3 = 17325),
                        # and that we only do this for top candidates,
                        # we can just recalculate the uncovered count for the affected positions
                        # or simply recalculate the whole thing if it's fast enough?
                        # No, full recalc is O(N) = 52k.
                        # Doing it 50*50=2500 times is 130M ops. Might be too slow in Python loop.
                        
                        # Let's optimize the swap check without committing.
                        # We want to know if net gain > 0.
                        # Net Gain = (Uncovered before) - (Uncovered after)
                        # Uncovered after = Uncovered before - (Covered by New & Not by Old) + (Uncovered by Old & Not by New)
                        
                        # Let's compute the actual delta using sets.
                        # old_pos_set = set(pos_old.tolist())
                        # new_pos_set = set(pos_new.tolist())
                        
                        # This is expensive.
                        
                        # Alternative:
                        # Just commit the swap, recalculate uncovered, and if it's not better, revert.
                        # Recalculating uncovered is O(N).
                        # If we do this 2500 times, it's 2500 * 52000 = 130M operations.
                        # In NumPy, `(cur_covered == 0).sum()` is very fast (vectorized C).
                        # 130M simple ops in C is < 0.1s.
                        # So we can afford to commit, check, and revert if needed.
                        
                        # Let's implement a "try swap" that commits, checks, and reverts.
                        
                        # Save state
                        old_classes = [c[:] for c in cur_classes]
                        old_covered = cur_covered.copy()
                        old_count = cur_count.copy()
                        old_used = set(cur_used)
                        old_uncovered = cur_uncovered
                        
                        # Perform swap on cur state
                        # (Already did the logic above, but let's clean it up)
                        
                        # Actually, the code above already modified cur_classes, cur_used, cur_count, cur_covered.
                        # Let's just check the result.
                        
                        new_uncovered = int((cur_covered == 0).sum())
                        
                        if new_uncovered < best_uncovered:
                            best_uncovered = new_uncovered
                            best_classes = [c[:] for c in cur_classes]
                            improved = True
                            found_swap = True
                            # Keep the state as it is (it's better)
                            cur_uncovered = new_uncovered
                        else:
                            # Revert
                            cur_classes = old_classes
                            cur_covered = old_covered
                            cur_count = old_count
                            cur_used = old_used
                            cur_uncovered = old_uncovered
                        
                        if found_swap:
                            break
                if found_swap:
                    break
            
        return best_classes, best_uncovered

    # Main Execution
    
    # 1. Greedy Initialization
    classes, covered, count, used = greedy_init(deadline - 20.0)
    best_uncovered = calculate_uncovered(covered)
    best_classes = [c[:] for c in classes]
    
    if time.time() > deadline - 10.0:
        return best_classes

    # 2. Local Search Polish
    best_classes, best_uncovered = local_search(
        best_classes, covered, count, used, deadline - 5.0
    )
    
    # 3. Randomized Restarts (if time permits)
    if time.time() < deadline - 15.0:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb best solution
            perturbed_classes = [c[:] for c in best_classes]
            if perturbed_classes:
                # Remove 1-2 random classes
                num_remove = random.randint(1, 2)
                indices_to_remove = random.sample(range(len(perturbed_classes)), min(num_remove, len(perturbed_classes)))
                for idx in sorted(indices_to_remove, reverse=True):
                    m, r = perturbed_classes[idx]
                    perturbed_classes.pop(idx)
            
            # Rebuild state
            p_covered = np.zeros(base, dtype=np.uint8)
            p_count = np.zeros(base, dtype=np.uint16)
            p_used = set()
            for m, r in perturbed_classes:
                m_idx = div_index[m]
                pos = positions[m_idx][r]
                p_covered[pos] = 1
                p_count[pos] += 1
                p_used.add(m)
            
            # Local search from perturbed state
            p_classes, p_uncovered = local_search(
                perturbed_classes, p_covered, p_count, p_used, deadline - 2.0
            )
            
            if p_uncovered < best_uncovered:
                best_uncovered = p_uncovered
                best_classes = p_classes

    return best_classes

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
    # Calculate score
    base = 51975
    union = 0
    for m, r in res:
        mask = 0
        x = r
        while x < base:
            mask |= (1 << x)
            x += m
        union |= mask
    uncovered = base - union.bit_count()
    print(f"Uncovered: {uncovered}")
