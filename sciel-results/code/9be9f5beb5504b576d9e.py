import time
import random

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    # Initial best: Powers of 2
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    def is_valid_addition(x, sums_by_card, current_n):
        # sums_by_card: dict {k: set(sums)}
        # current_n: current size of the set
        # Check if adding x creates a conflict
        
        # We iterate over existing cardinalities k
        # New sums for cardinality k+1 come from sums in k + x
        # These new sums must not be in sums_by_card[j] for any j != k+1
        
        for k in range(current_n + 1):
            if k not in sums_by_card:
                continue
            existing_sums_k = sums_by_card[k]
            new_sums = {s + x for s in existing_sums_k}
            
            # Check against all other cardinalities
            for j in range(current_n + 2): # Max new cardinality is current_n + 1
                if j == k + 1:
                    continue
                if j not in sums_by_card:
                    continue
                # Check intersection
                # Optimization: if len(new_sums) < len(sums_by_card[j]), iterate new_sums
                # else iterate sums_by_card[j]
                # But sets are small enough, just use &
                if new_sums & sums_by_card[j]:
                    return False
        return True

    def greedy_construct(randomize_strategy=True):
        S = []
        sums_by_card = {0: {0}}
        current_max_card = 0
        
        used = set()
        
        while True:
            valid_candidates = []
            # We can limit search to 1..256
            # To speed up, we can iterate 1..256 and check
            # But checking is O(N^2 * SumSize).
            # Let's just iterate 1..256
            
            for x in range(1, 257):
                if x in used:
                    continue
                if is_valid_addition(x, sums_by_card, len(S)):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            if randomize_strategy:
                # 80% smallest, 20% random
                if random.random() < 0.8:
                    x = min(valid_candidates)
                else:
                    x = random.choice(valid_candidates)
            else:
                x = min(valid_candidates)
            
            # Add x
            S.append(x)
            used.add(x)
            
            # Update sums_by_card
            # New sums for k+1 come from old sums of k + x
            # We must update from high k to low k to avoid using updated values in the same step?
            # No, sums_by_card[k] is the set of sums for subsets of size k from the PREVIOUS set.
            # When we add x, the new set of sums for size k+1 is old_sums[k+1] U {s+x | s in old_sums[k]}
            # We need to store the old state or update carefully.
            
            # Let's create a new dict
            new_sums_by_card = {k: set(v) for k, v in sums_by_card.items()}
            
            for k in range(len(S) - 1, -1, -1):
                if k in new_sums_by_card:
                    # Add new sums to k+1
                    target_k = k + 1
                    if target_k not in new_sums_by_card:
                        new_sums_by_card[target_k] = set()
                    
                    # s + x for s in sums_by_card[k] (OLD sums)
                    # Note: we must use the OLD sums_by_card[k], not new_sums_by_card[k]
                    # Because subsets of size k in the NEW set either include x (size k-1 in old) or not (size k in old).
                    # The sums NOT including x are already in new_sums_by_card[k] (copied from old).
                    # The sums INCLUDING x are {s+x | s in old_sums[k-1]}.
                    # Wait, my update logic above was slightly off.
                    
                    # Let's re-verify.
                    # Old set S_old. Sums S_old_k.
                    # New set S_new = S_old U {x}.
                    # Sums S_new_k = {sums of k-subsets of S_new}.
                    # A k-subset of S_new either contains x or not.
                    # If not: sum is in S_old_k.
                    # If yes: sum = x + sum of (k-1)-subset of S_old. So sum is in {x + s | s in S_old_{k-1}}.
                    # So S_new_k = S_old_k U {x + s | s in S_old_{k-1}}.
                    
                    # In my loop:
                    # I am iterating k.
                    # I want to update new_sums_by_card[k+1] using old_sums_by_card[k].
                    # This matches: S_new_{k+1} gets {x+s | s in S_old_k}.
                    
                    for s in sums_by_card.get(k, set()):
                        new_sums_by_card[target_k].add(s + x)
            
            sums_by_card = new_sums_by_card

        return S

    # Run multiple random restarts
    while time.time() < deadline:
        S = greedy_construct(randomize_strategy=True)
        if len(S) > len(best_set):
            best_set = S
            # If we reach a theoretical high, we might stop? 
            # Theoretical max for 1..256?
            # With N=15, max sum ~ 15*256 = 3840.
            # Number of subsets 2^15 = 32768.
            # Sums must be disjoint per cardinality.
            # Total distinct sums needed = sum_{k=0}^N C(N,k) = 2^N.
            # Max possible sum is ~ 4000.
            # So 2^N <= 4000 is not strictly required because sums can repeat within cardinality.
            # But the total number of distinct pairs (sum, cardinality) is 2^N.
            # The range of sums is 0..4000.
            # So we need 2^N distinct (sum, k) pairs.
            # Since k is determined, we just need the sets S_k to be disjoint.
            # Total distinct sums used = sum |S_k|.
            # This must be <= 4000.
            # For N=12, 2^12 = 4096. This is close to 4000.
            # So N=12 is likely the limit. N=13 (8192 subsets) would require > 4000 distinct sums on average?
            # Not necessarily, if many collisions within same k.
            # But S_k must be disjoint from S_j.
            # So total distinct sums = sum |S_k| <= 4000.
            # If N=13, we have 14 cardinalities.
            # It's tight.
            # Let's see if we can find 12.
        
        # If we found 13+, we can probably stop or keep trying for 14.
        if len(best_set) >= 13:
            # Check if we have time for one more
            if time.time() > deadline - 5:
                break

    return best_set
