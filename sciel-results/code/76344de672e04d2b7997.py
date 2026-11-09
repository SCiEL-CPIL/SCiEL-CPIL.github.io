import random
import time
import itertools

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_sidon_subsets_of_size(s, k):
    """Returns list of tuples that are Sidon subsets of size k. Short circuits if we only need one? No, we might want to break multiple."""
    subs = []
    for combo in itertools.combinations(s, k):
        if is_sidon(combo):
            subs.append(combo)
    return subs

def max_sidon_size(s):
    # Check 5, 4, 3
    # If we find a 5, return 5 (bad)
    # If we find a 4, return 4
    # If we find a 3, return 3
    # Else 2
    
    # Optimization: Check 5 first? 
    # For n=12, 5-Sidon is rare in dense sets. 
    # But if we have a 5-Sidon, we definitely have 4-Sidon.
    # So if we find a 4-Sidon, we don't need to check 5 to know it's >= 4.
    # But we want to know if it's 5 to prioritize breaking it?
    # Breaking a 5 is harder.
    # Let's just check 4 first. If none, check 3.
    # Wait, if there is a 5, there is a 4.
    # So if `find_sidon_subsets_of_size(s, 4)` is not empty, max size >= 4.
    # Do we need to distinguish 4 from 5?
    # The score is the max size. 5 is worse than 4.
    # So if we have a 5, we should try to break it.
    # However, breaking a 5 usually breaks all 4s inside it.
    # So targeting 4s is sufficient to reduce 5s.
    
    # Let's check 4.
    # If any 4-Sidon exists, return 4. (We don't care if it's 5, we just know it's >=4).
    # Actually, if it's 5, we want to know? 
    # If we return 4, and it's actually 5, our score is 5.
    # If we return 4, and it's actually 4, our score is 4.
    # We want to minimize the score.
    # So we should check if 5 exists.
    
    # But checking 5 is more expensive.
    # Let's assume for now that 4 is the bottleneck.
    # If we eliminate 4s, we eliminate 5s.
    # So if we find a 4-Sidon, we know max >= 4.
    # We can just return 4 as a proxy for "bad".
    # If we find no 4-Sidon, we check 3.
    # If we find a 3-Sidon, return 3.
    # Else return 2.
    
    # This proxy works because:
    # If max=5, count(4)>0. We return 4. (Score 5 vs 4. We underestimate the badness).
    # If max=4, count(4)>0. We return 4. (Score 4).
    # If max=3, count(4)=0. We check 3. Return 3. (Score 3).
    # If max=2, count(4)=0, count(3)=0. Return 2. (Score 2).
    
    # The problem: We might prefer a set with max=4 over a set with max=5, but both return 4.
    # This is fine for local search: we just want to get to 3 or 2.
    # Once we are at 4, we try to get to 3.
    # If we are stuck at 4, we might be oscillating between 4 and 5?
    # Unlikely.
    
    if find_sidon_subsets_of_size(s, 4):
        return 4
    if find_sidon_subsets_of_size(s, 3):
        return 3
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_set = None
    best_score = 10 # Max possible is 12, but 4 is likely min for random
    
    # Initial candidate: AP
    cand_ap = list(range(12))
    score_ap = max_sidon_size(cand_ap)
    best_set = cand_ap
    best_score = score_ap
    
    # Local Search
    current_set = set(random.sample(range(129), 12))
    current_score = max_sidon_size(list(current_set))
    
    if current_score < best_score:
        best_set = list(current_set)
        best_score = current_score
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        if best_score == 2:
            break
            
        # Targeted Mutation:
        # If current_score >= 4, find a Sidon 4-subset and break it.
        # If current_score == 3, find a Sidon 3-subset and break it?
        # Breaking a 3-subset means making one of its pairs sum to another pair's sum.
        
        if current_score >= 4:
            # Find one Sidon 4-subset
            sidon_4s = find_sidon_subsets_of_size(list(current_set), 4)
            if sidon_4s:
                target_subset = sidon_4s[0]
                # Pick an element from target_subset to move
                idx_to_move = random.choice(range(4))
                old_val = target_subset[idx_to_move]
                
                # Find index of old_val in current_set
                list_cs = list(current_set)
                pos_in_list = list_cs.index(old_val)
                
                # Try to move to a value that breaks the Sidon property of the new subset
                # New subset will be (target_subset \ {old_val}) U {new_val}
                # We want this new subset to NOT be Sidon.
                
                # Try random new values
                found_good_move = False
                for _ in range(50): # Try 50 random new values
                    new_val = random.randint(0, 128)
                    if new_val in current_set and new_val != old_val:
                        continue
                    
                    # Construct new set
                    new_set = set(list_cs)
                    new_set.remove(old_val)
                    new_set.add(new_val)
                    
                    # Check if the specific subset is broken
                    # The subset is target_subset with old_val replaced by new_val
                    new_subset = list(target_subset)
                    new_subset[idx_to_move] = new_val
                    
                    # If new_subset is NOT Sidon, this move is good for this specific subset
                    # But we need to check if the whole set's score improves
                    # Actually, if we break this 4-subset, the score might drop to 3 or stay 4 (if another 4-subset exists).
                    # So we should evaluate the whole set.
                    
                    new_score = max_sidon_size(list(new_set))
                    if new_score < current_score:
                        current_set = new_set
                        current_score = new_score
                        found_good_move = True
                        break
                    elif new_score == current_score and random.random() < 0.1:
                        # Accept equal score with some probability (Simulated Annealing lite)
                        current_set = new_set
                        # current_score unchanged
                        found_good_move = True
                        break
                        
                if found_good_move:
                    if current_score < best_score:
                        best_score = current_score
                        best_set = list(current_set)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
        else:
            # current_score == 3
            # Find a Sidon 3-subset and break it
            sidon_3s = find_sidon_subsets_of_size(list(current_set), 3)
            if sidon_3s:
                target_subset = sidon_3s[0]
                idx_to_move = random.choice(range(3))
                old_val = target_subset[idx_to_move]
                list_cs = list(current_set)
                pos_in_list = list_cs.index(old_val)
                
                found_good_move = False
                for _ in range(50):
                    new_val = random.randint(0, 128)
                    if new_val in current_set and new_val != old_val:
                        continue
                    
                    new_set = set(list_cs)
                    new_set.remove(old_val)
                    new_set.add(new_val)
                    
                    new_score = max_sidon_size(list(new_set))
                    if new_score < current_score:
                        current_set = new_set
                        current_score = new_score
                        found_good_move = True
                        break
                    elif new_score == current_score and random.random() < 0.1:
                        current_set = new_set
                        found_good_move = True
                        break
                        
                if found_good_move:
                    if current_score < best_score:
                        best_score = current_score
                        best_set = list(current_set)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
        # Random Restart if stuck
        if no_improve_count > max_no_improve:
            current_set = set(random.sample(range(129), 12))
            current_score = max_sidon_size(list(current_set))
            if current_score < best_score:
                best_score = current_score
                best_set = list(current_set)
            no_improve_count = 0
            
    return best_set
