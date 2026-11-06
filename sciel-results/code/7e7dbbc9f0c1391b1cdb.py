import random
import time

def construct():
    n = 128
    target_start = 128
    target_end = 256
    
    # Initial set: Seed
    A = set(range(64, 129))
    
    def build_counts(A_set):
        counts = [0] * (2 * n + 1)
        for a in A_set:
            for b in A_set:
                if a <= b:
                    counts[a + b] += 1
        return counts

    counts = build_counts(A)
    
    def is_valid(counts):
        for s in range(target_start, target_end + 1):
            if counts[s] == 0:
                return False
        return True

    def get_score(counts):
        # Primary: minimize peak
        # Secondary: minimize sum of squares (balance)
        peak = max(counts[target_start:target_end + 1])
        sq = sum(c * c for c in counts[target_start:target_end + 1])
        return (peak, sq)

    current_score = get_score(counts)
    
    start_time = time.time()
    timeout = 55.0
    
    # Helper to apply a move (add/remove) and return delta counts or just update
    # We will use in-place updates for speed.

    def try_remove(a, A_set, counts, current_score):
        """
        Tries to remove a from A_set.
        Returns True if removal was accepted (score improved or equal), False otherwise.
        """
        # Tentatively remove a
        A_set.remove(a)
        
        # Update counts: subtract pairs (a, b) for b in A_set, and (a, a)
        for b in A_set:
            counts[a + b] -= 1
        counts[2 * a] -= 1
        
        if is_valid(counts):
            new_score = get_score(counts)
            if new_score <= current_score:
                # Accept
                return True, new_score
            else:
                # Restore
                A_set.add(a)
                for b in A_set:
                    counts[a + b] += 1
                counts[2 * a] += 1
                return False, current_score
        else:
            # Restore
            A_set.add(a)
            for b in A_set:
                counts[a + b] += 1
            counts[2 * a] += 1
            return False, current_score

    def try_swap(a, b, A_set, counts, current_score):
        """
        Tries to swap a in A_set with b not in A_set.
        Returns True if swap was accepted, False otherwise.
        """
        if a == b:
            return False, current_score
            
        # Remove a
        A_set.remove(a)
        for c in A_set:
            counts[a + c] -= 1
        counts[2 * a] -= 1
        
        # Add b
        for c in A_set:
            counts[b + c] += 1
        counts[2 * b] += 1
        A_set.add(b)
        
        if is_valid(counts):
            new_score = get_score(counts)
            if new_score <= current_score:
                # Accept
                return True, new_score
            else:
                # Restore
                A_set.remove(b)
                for c in A_set:
                    counts[b + c] -= 1
                counts[2 * b] -= 1
                    
                A_set.add(a)
                for c in A_set:
                    counts[a + c] += 1
                counts[2 * a] += 1
                return False, current_score
        else:
            # Restore
            A_set.remove(b)
            for c in A_set:
                counts[b + c] -= 1
            counts[2 * b] -= 1
                    
            A_set.add(a)
            for c in A_set:
                counts[a + c] += 1
            counts[2 * a] += 1
            return False, current_score

    stuck_counter = 0
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Phase 1: Removals
        A_list = list(A)
        random.shuffle(A_list)
        
        for a in A_list:
            if time.time() - start_time > timeout:
                break
            
            success, new_score = try_remove(a, A, counts, current_score)
            if success:
                current_score = new_score
                improved = True
        
        # Phase 2: Swaps
        if not improved:
            swapped = False
            for _ in range(100): # Try more swaps
                if time.time() - start_time > timeout:
                    break
                if len(A) < 1:
                    break
                a = random.choice(list(A))
                candidates = [x for x in range(n + 1) if x not in A]
                if not candidates:
                    break
                b = random.choice(candidates)
                
                success, new_score = try_swap(a, b, A, counts, current_score)
                if success:
                    current_score = new_score
                    swapped = True
                    improved = True
                    # Don't break, keep looking for more improvements
                    # But restart the swap loop logic? No, just continue.
            
            if not swapped:
                # Stuck
                stuck_counter += 1
                if stuck_counter > 5:
                    # Try a kick: Add a random element, then try to remove
                    candidates = [x for x in range(n + 1) if x not in A]
                    if candidates:
                        kick_elem = random.choice(candidates)
                        # Add kick_elem
                        A.add(kick_elem)
                        for c in A:
                            if c != kick_elem:
                                counts[kick_elem + c] += 1
                        counts[2 * kick_elem] += 1
                        
                        # Now try to remove elements to get back to valid state with better score
                        # The set is now larger, so we definitely have more counts.
                        # We need to remove elements to restore validity and minimize score.
                        # Run a few removal passes.
                        for _ in range(3):
                            A_list_kick = list(A)
                            random.shuffle(A_list_kick)
                            for a_k in A_list_kick:
                                if time.time() - start_time > timeout:
                                    break
                                success, new_score = try_remove(a_k, A, counts, current_score)
                                if success:
                                    current_score = new_score
                                    improved = True
                        stuck_counter = 0
                    else:
                        break
                else:
                    # Just break if we've been stuck a few times but not too many?
                    # Or continue with more swaps?
                    # Let's break to avoid wasting time if truly stuck.
                    if stuck_counter > 20:
                        break
        else:
            stuck_counter = 0

    return list(A)
