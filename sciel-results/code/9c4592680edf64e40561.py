import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit (evaluator kills at 150s, but keep tight)
    
    # 1. Greedy Initialization (Baseline)
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # 2. Fast Objective Calculation
    # Returns (scope, penalty, used_diffs_set)
    # Penalty is number of duplicate differences.
    def calc_obj(rows):
        diffs = set()
        penalty = 0
        max_val = 0
        for row in rows:
            # Update max_val
            if row:
                if row[-1] > max_val:
                    max_val = row[-1]
            # Calculate differences
            for i in range(len(row)):
                ai = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ai
                    if d in diffs:
                        penalty += 1
                    else:
                        diffs.add(d)
        return max_val, penalty, diffs

    # 3. Error Repair Move
    # Given a row index, position index, and current used diffs (excluding this row's diffs if we are modifying it? 
    # Actually, easier: 
    # 1. Remove current row's diffs from global set.
    # 2. Try to change row[r_idx][p_idx] to a new value.
    # 3. Check if new row's diffs collide with (global_set - old_row_diffs).
    # 4. If valid, accept.
    
    def try_repair_and_reduce(current_rows, current_used, r_idx, p_idx, target_scope):
        # Remove current row's diffs from the global set to check against others
        row = current_rows[r_idx]
        old_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                old_diffs.add(row[j] - row[i])
        
        # We need to check against (current_used - old_diffs)
        # But current_used includes old_diffs.
        # Let's just compute the "other_diffs" set once outside or handle carefully.
        # For efficiency in a tight loop, we can just check the new diffs against current_used,
        # but we must ignore collisions with itself.
        
        # Actually, the simplest robust way for a single move:
        # 1. Compute new_diffs for the modified row.
        # 2. Check if any new_diff is in (current_used - old_diffs).
        
        # Let's optimize:
        # We know current_used.
        # We compute new_diffs.
        # Collision exists if d in new_diffs AND d in current_used AND d not in old_diffs.
        
        min_val = row[p_idx - 1] + 1
        # We want to minimize scope, so we prefer smaller values.
        # But we must ensure strict increase.
        # If p_idx == k, we can go up to target_scope or current_val + something.
        # Let's try a range of values.
        
        # Strategy: Try values starting from min_val upwards, but maybe jump around.
        # To allow "error repair", we look for a value that creates NO new collisions with OTHER rows.
        
        # Pre-compute "forbidden" diffs: diffs used by other rows.
        # This is expensive to compute every time. 
        # Alternative: Just check against current_used.
        # If a diff d is in new_diffs:
        #   If d is in old_diffs, it's fine (it was there before).
        #   If d is NOT in old_diffs, it must NOT be in current_used.
        
        # Let's try a few random values and some greedy small values.
        
        # 1. Try to reduce scope: try values in [min_val, min(target_scope, current_val)]
        # 2. Try to fix errors: try random values in [min_val, target_scope]
        
        # For this specific move, let's just try to find ANY valid value in [min_val, target_scope]
        # that minimizes the value (to reduce scope) or simply fixes a collision.
        
        # To keep it fast, we try a small number of candidates.
        
        candidates = []
        # Greedy: smallest possible
        if min_val <= target_scope:
            candidates.append(min_val)
        # Random
        if min_val < target_scope:
            for _ in range(5):
                candidates.append(random.randint(min_val, target_scope))
        # Current value (to check if it's already optimal/valid)
        candidates.append(row[p_idx])
        
        best_candidate = None
        best_candidate_val = float('inf')
        
        for val in candidates:
            if val < min_val or val > target_scope:
                continue
            if val == row[p_idx]:
                # Check if current is valid? It is, by definition of state, unless we are in an invalid state.
                # If we are in an invalid state, we want to move away.
                # If we are in a valid state, we want to reduce scope.
                # Let's just evaluate it.
                pass
            
            # Construct new row
            new_row = row[:]
            new_row[p_idx] = val
            
            # Check strict increase
            if p_idx > 0 and new_row[p_idx] <= new_row[p_idx-1]:
                continue
            if p_idx < len(new_row)-1 and new_row[p_idx] >= new_row[p_idx+1]:
                continue
            
            # Calculate new diffs
            new_diffs = set()
            for i in range(len(new_row)):
                ai = new_row[i]
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - ai
                    new_diffs.add(d)
            
            # Check collisions
            # A collision is a diff in new_diffs that is in current_used but NOT in old_diffs.
            # Note: if a diff is in new_diffs and old_diffs, it's "reused" within the row, which is fine 
            # as long as it doesn't collide with OTHER rows.
            # Wait, the condition is: ALL differences between two entries of the same row are distinct OVER ALL ROWS.
            # This means:
            # 1. Within a row, diffs are distinct? No, "all differences ... are distinct over all rows".
            #    This implies the set of ALL diffs from ALL rows must be a set (no duplicates).
            #    So if row 1 has diff 5 and row 2 has diff 5, it's invalid.
            #    If row 1 has diff 5 twice? A row has distinct entries, so diffs a_j - a_i for i<j.
            #    Can a row have duplicate diffs? e.g. 0, 1, 2 -> diffs 1, 2, 1. Yes.
            #    But the global set must be distinct. So if row 1 has {1, 2, 1}, the global set gets 1 and 2.
            #    The "distinct over all rows" usually means the multiset of all pairwise differences across all rows has no duplicates.
            #    So if row 1 produces diff 1 twice, that's two 1s in the global multiset?
            #    "all differences ... are distinct" -> The set of values must be unique.
            #    So if any diff appears more than once in the entire collection, it's invalid.
            
            # So, for the new row:
            # 1. Any diff d in new_diffs that is ALSO in old_diffs?
            #    If old_diffs had 1 and new_diffs has 1, and 1 was not in other rows, it's fine.
            #    But if old_diffs had 1 (meaning it contributed to the global count) and we change it, 
            #    we remove the old contribution and add the new.
            
            # Let's simplify the check:
            # Global Validity Condition:
            # Let D be the multiset of all diffs from all rows.
            # We require all elements in D to be distinct.
            # 
            # Current State:
            # We have `current_used` which is a SET. This implies the current state is VALID (no duplicates).
            # If we are in an INVALID state (penalty > 0), `current_used` is just the set of unique diffs, 
            # but we've lost count info.
            # 
            # To handle invalid states properly, we should track counts or just accept that 
            # we are doing local search on a penalty landscape.
            # 
            # Let's stick to the penalty landscape approach.
            # Penalty = number of duplicate diffs.
            # 
            # To check if a move reduces penalty:
            # We need to know how many times each diff appears.
            # This is heavy.
            
            # Alternative: Only move if the new row's diffs do NOT collide with OTHER rows' diffs.
            # And within the new row, diffs are distinct? No, within a row, diffs can repeat?
            # "all differences ... are distinct over all rows"
            # If row is [0, 1, 2], diffs are 1, 2, 1. The values are 1 and 2.
            # If the requirement is that the SET of all difference values is distinct, then [0,1,2] is fine 
            # provided no other row has 1 or 2.
            # But if the requirement is that the MULTISET of all differences has distinct elements, 
            # then [0,1,2] is INVALID because it produces two 1s.
            # 
            # Standard "Difference Triangle" / "Golomb Ruler" generalization:
            # Usually, "distinct differences" means the set of all pairwise differences is a set (no duplicates).
            # This implies that within a single row, differences must also be distinct?
            # If a row has duplicate differences, then that value appears at least twice in the global collection.
            # So, YES, within a row, all pairwise differences must be distinct.
            # This is the definition of a Golomb Ruler.
            # So each row must be a Golomb Ruler.
            # AND the sets of differences from different rows must be disjoint.
            
            # So:
            # 1. New row must be a valid Golomb Ruler (all internal diffs distinct).
            # 2. New row's diffs must be disjoint from all other rows' diffs.
            
            # Check 1: Internal Golomb
            if len(new_diffs) != len(new_row) * (len(new_row) - 1) // 2:
                continue
            
            # Check 2: Disjoint from others
            # We need to know the diffs of OTHER rows.
            # We have `current_used` (all diffs).
            # We have `old_diffs` (current row's diffs).
            # Other diffs = current_used - old_diffs (as sets, assuming current state is valid).
            # If current state is invalid, this logic breaks.
            # But if we start from a valid greedy solution and only make moves that preserve validity 
            # (or move towards it), we can assume we are mostly in valid states or near-valid.
            
            # To be safe and simple:
            # If we assume current state is VALID (penalty 0):
            # Then `current_used` is the disjoint union of diffs from all rows.
            # Other_diffs = current_used - old_diffs.
            # If new_diffs & Other_diffs is empty, the move is valid.
            
            # If we are in an INVALID state, we can't easily compute this without counts.
            # Let's enforce that we only accept moves that result in a VALID state 
            # (new_diffs disjoint from current_used - old_diffs) 
            # AND reduce scope.
            # This restricts us to valid states.
            
            # If we are stuck in an invalid state, we can't escape this way.
            # But we start valid. If we only make valid moves, we stay valid.
            # So we can just search the space of valid configurations.
            
            # Is the space of valid configurations connected?
            # Probably not fully, but local search might work.
            
            # Check disjointness
            # Note: old_diffs might not be a subset of current_used if state is invalid.
            # But if we maintain validity, it is.
            
            # Let's assume we are in a valid state.
            # other_diffs = current_used - old_diffs
            # if new_diffs & other_diffs:
            #     continue
            
            # To avoid set difference every time, we can just check:
            # for d in new_diffs:
            #    if d in current_used and d not in old_diffs:
            #        break
            
            valid_move = True
            for d in new_diffs:
                if d in current_used:
                    if d not in old_diffs:
                        valid_move = False
                        break
            
            if not valid_move:
                continue
            
            # If valid, accept if it reduces scope
            if val < best_candidate_val:
                best_candidate_val = val
                best_candidate = val
        
        if best_candidate is not None:
            current_rows[r_idx][p_idx] = best_candidate
            return True
        return False

    # Main Loop
    rows = greedy_init()
    scope, penalty, used_diffs = calc_obj(rows)
    best_rows = [r[:] for r in rows]
    best_scope = scope
    
    # Population based search
    # We will maintain a list of "islands" or just do multi-start?
    # Given time limit, simple multi-start with local search on each might be better.
    # But let's try a population of 10 solutions.
    
    pop_size = 10
    population = []
    
    # Initial population: 1 greedy, 9 random valid-ish?
    # Random valid generation is hard.
    # Let's just copy the greedy one and perturb.
    
    for i in range(pop_size):
        if i == 0:
            p_rows = [r[:] for r in rows]
        else:
            # Perturb the greedy solution
            p_rows = [r[:] for r in rows]
            # Randomly change some values
            for _ in range(5):
                r_idx = random.randint(0, n_rows - 1)
                p_idx = random.randint(1, k)
                # Just randomize value within bounds
                min_v = p_rows[r_idx][p_idx-1] + 1
                max_v = p_rows[r_idx][p_idx+1] - 1 if p_idx < k else p_rows[r_idx][p_idx] + 10
                if min_v <= max_v:
                    p_rows[r_idx][p_idx] = random.randint(min_v, max_v)
        
        p_scope, p_penalty, p_used = calc_obj(p_rows)
        population.append({
            'rows': p_rows,
            'scope': p_scope,
            'penalty': p_penalty,
            'used': p_used
        })
        
        if p_penalty == 0 and p_scope < best_scope:
            best_scope = p_scope
            best_rows = [r[:] for r in p_rows]

    # Local Search on Population
    # Each iteration, pick a random individual, perform a move, update.
    
    while time.time() < deadline:
        # Pick random individual
        idx = random.randint(0, pop_size - 1)
        ind = population[idx]
        
        if ind['penalty'] > 0:
            # Try to fix error
            # Pick a random row and position
            r_idx = random.randint(0, n_rows - 1)
            p_idx = random.randint(1, k)
            
            # Try to repair
            # We need to find a value that makes the row a Golomb ruler 
            # and disjoint from others.
            # This is hard to do directly.
            # Let's just try random values and see if penalty drops.
            
            old_val = ind['rows'][r_idx][p_idx]
            min_val = ind['rows'][r_idx][p_idx - 1] + 1
            max_val = ind['rows'][r_idx][p_idx + 1] - 1 if p_idx < k else ind['rows'][r_idx][p_idx] + 20
            
            if min_val > max_val:
                continue
                
            new_val = random.randint(min_val, max_val)
            
            ind['rows'][r_idx][p_idx] = new_val
            new_scope, new_penalty, new_used = calc_obj(ind['rows'])
            
            # Accept if penalty reduced, or same penalty and scope reduced
            if new_penalty < ind['penalty'] or (new_penalty == ind['penalty'] and new_scope < ind['scope']):
                ind['rows'] = [r[:] for r in ind['rows']]
                ind['scope'] = new_scope
                ind['penalty'] = new_penalty
                ind['used'] = new_used
                
                if new_penalty == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in ind['rows']]
            else:
                # Revert
                ind['rows'][r_idx][p_idx] = old_val
                # Recalc if needed? We didn't change it back in the object yet, 
                # but we modified the list in place. 
                # We need to restore the object state.
                # Since we modified ind['rows'] in place, we should restore it.
                # But we don't have the old used_diffs easily if we don't recompute.
                # Let's just recompute if we revert? No, too slow.
                # Better: Don't modify in place. Create a copy.
                
                # This indicates a flaw in the in-place modification approach for reverts.
                # Let's refactor to use copies for trial moves.
                
        else:
            # Valid state: Try to reduce scope
            r_idx = random.randint(0, n_rows - 1)
            p_idx = random.randint(1, k)
            
            row = ind['rows'][r_idx]
            old_val = row[p_idx]
            min_val = row[p_idx - 1] + 1
            # Try to reduce, so look at [min_val, old_val]
            if min_val < old_val:
                # Try a few candidates
                candidates = [min_val, old_val - 1, random.randint(min_val, old_val)]
                for new_val in candidates:
                    if new_val < min_val or new_val > old_val:
                        continue
                    
                    # Trial move
                    trial_row = row[:]
                    trial_row[p_idx] = new_val
                    
                    # Check Golomb property
                    trial_diffs = set()
                    for i in range(len(trial_row)):
                        for j in range(i + 1, len(trial_row)):
                            d = trial_row[j] - trial_row[i]
                            trial_diffs.add(d)
                    
                    if len(trial_diffs) != len(trial_row) * (len(trial_row) - 1) // 2:
                        continue
                    
                    # Check disjointness
                    # old_diffs of this row
                    old_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            old_diffs.add(row[j] - row[i])
                    
                    valid = True
                    for d in trial_diffs:
                        if d in ind['used']:
                            if d not in old_diffs:
                                valid = False
                                break
                    
                    if valid:
                        # Update
                        ind['rows'][r_idx] = trial_row
                        ind['scope'] = min(ind['scope'], new_val) # Actually max of all entries
                        # Recalc scope properly
                        max_v = 0
                        for r in ind['rows']:
                            if r[-1] > max_v:
                                max_v = r[-1]
                        ind['scope'] = max_v
                        
                        # Update used diffs
                        # Remove old_diffs, add trial_diffs
                        # Since they are disjoint from others, and old_diffs were in used,
                        # we can just do:
                        # used = (used - old_diffs) | trial_diffs
                        # But we need to be careful if old_diffs and trial_diffs overlap?
                        # They are both from the same row, but different values.
                        # It's possible a diff value is in both old and new?
                        # e.g. row [0, 2, 4] -> diffs 2, 4, 2 (invalid Golomb)
                        # Since we checked Golomb, diffs are distinct within row.
                        # So old_diffs and trial_diffs are sets.
                        # Can a value be in both?
                        # Yes, e.g. row [0, 1, 3] (diffs 1, 3, 2) -> change to [0, 2, 3] (diffs 2, 3, 1).
                        # 1, 2, 3 are in both.
                        # So we can't just subtract.
                        # We must recompute used_diffs from scratch for the whole population?
                        # No, just for this individual?
                        # The 'used' set in the individual is the union of all rows' diffs.
                        # If we change one row, the new used set is (old_used - old_row_diffs) | new_row_diffs.
                        # This is correct IF old_row_diffs is a subset of old_used (which it is).
                        
                        ind['used'] = (ind['used'] - old_diffs) | trial_diffs
                        
                        if ind['scope'] < best_scope:
                            best_scope = ind['scope']
                            best_rows = [r[:] for r in ind['rows']]
                        
                        break # Accept first valid move found
            
    return best_rows
