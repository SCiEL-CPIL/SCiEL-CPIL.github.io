import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    def can_add_to_set(x, elements, sums, diffs_other):
        """Check if x can be added to a Sidon set with given constraints."""
        if x + x in sums:
            return False
        for a in elements:
            if x + a in sums:
                return False
        # Check difference conflicts with other set
        for a in elements:
            if (x - a) in diffs_other or (a - x) in diffs_other:
                return False
        return True
    
    def add_to_set(x, elements, sums, diffs):
        """Add x to a Sidon set, updating sums and diffs."""
        elements.append(x)
        sums.add(x + x)
        for a in elements[:-1]:
            sums.add(x + a)
            diffs.add(x - a)
            diffs.add(a - x)
    
    def try_add_anywhere(x, A, B, sums_A, sums_B, diffs_A, diffs_B):
        """Try to add x to either A or B. Returns True if added."""
        valid_A = can_add_to_set(x, A, sums_A, diffs_B)
        valid_B = can_add_to_set(x, B, sums_B, diffs_A)
        
        if valid_A and valid_B:
            # Balance: prefer the smaller set, with slight randomization
            if len(A) < len(B) - 1:
                side = 'A'
            elif len(B) < len(A) - 1:
                side = 'B'
            else:
                # When balanced, try to minimize future conflicts
                # Count how many new differences would be created
                new_diffs_A = 0
                for a in A:
                    if (x - a) not in diffs_A:
                        new_diffs_A += 1
                new_diffs_B = 0
                for b in B:
                    if (x - b) not in diffs_B:
                        new_diffs_B += 1
                if new_diffs_A < new_diffs_B - 1:
                    side = 'A'
                elif new_diffs_B < new_diffs_A - 1:
                    side = 'B'
                else:
                    side = random.choice(['A', 'B'])
        elif valid_A:
            side = 'A'
        elif valid_B:
            side = 'B'
        else:
            return False
        
        if side == 'A':
            add_to_set(x, A, sums_A, diffs_A)
        else:
            add_to_set(x, B, sums_B, diffs_B)
        return True
    
    best_size = 0
    best_A = []
    best_B = []
    
    max_restarts = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 1.0:
            break
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Generate candidate ordering
        strategy = random.random()
        if strategy < 0.3:
            # Pure random
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        elif strategy < 0.6:
            # Biased toward smaller numbers (they tend to have more room)
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            candidates.sort(key=lambda x: x * 0.3 + random.random() * 100)
        elif strategy < 0.8:
            # Biased toward middle
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            candidates.sort(key=lambda x: abs(x - 128) * 0.5 + random.random() * 150)
        else:
            # Greedy: try to add smallest valid first
            candidates = list(range(1, 257))
        
        # Main construction phase
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            try_add_anywhere(x, A, B, sums_A, sums_B, diffs_A, diffs_B)
        
        # Local improvement: multiple passes trying to add remaining elements
        if time.time() < deadline - 5.0:
            remaining = set(range(1, 257)) - set(A) - set(B)
            
            # Multiple passes with different orderings
            for pass_num in range(3):
                if time.time() > deadline - 5.0:
                    break
                if pass_num == 0:
                    order = sorted(remaining, key=lambda v: random.random())
                elif pass_num == 1:
                    order = sorted(remaining)
                else:
                    order = sorted(remaining, reverse=True)
                
                added_any = False
                for x in order:
                    if time.time() > deadline - 5.0:
                        break
                    if try_add_anywhere(x, A, B, sums_A, sums_B, diffs_A, diffs_B):
                        added_any = True
                
                if not added_any:
                    break
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = list(A)
            best_B = list(B)
    
    # Final attempt: try to squeeze out one more element if possible
    if time.time() < deadline - 1.0 and best_A and best_B:
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for a in best_A:
            sums_A.add(a + a)
        for i in range(len(best_A)):
            for j in range(i + 1, len(best_A)):
                sums_A.add(best_A[i] + best_A[j])
                diffs_A.add(best_A[i] - best_A[j])
                diffs_A.add(best_A[j] - best_A[i])
        
        for b in best_B:
            sums_B.add(b + b)
        for i in range(len(best_B)):
            for j in range(i + 1, len(best_B)):
                sums_B.add(best_B[i] + best_B[j])
                diffs_B.add(best_B[i] - best_B[j])
                diffs_B.add(best_B[j] - best_B[i])
        
        for x in range(1, 257):
            if x in best_A or x in best_B:
                continue
            if can_add_to_set(x, best_A, sums_A, diffs_B):
                best_A.append(x)
                break
            elif can_add_to_set(x, best_B, sums_B, diffs_A):
                best_B.append(x)
                break
    
    return [best_A, best_B]
