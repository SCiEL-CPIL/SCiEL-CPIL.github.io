import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s/90s limit
    n = 256
    max_possible = 32  # Theoretical upper bound derived from difference count
    
    class SetState:
        __slots__ = ['elements', 'diffs']
        
        def __init__(self):
            self.elements = []
            self.diffs = set()
            
        def try_add(self, x, other_diffs):
            # Check if x can be added to this set
            # Constraints:
            # 1. |x - e| must not be in self.diffs (Sidon/Distinct Diffs for this set)
            # 2. |x - e| must not be in other_diffs (Disjoint Diffs)
            for e in self.elements:
                d = abs(x - e)
                if d in self.diffs or d in other_diffs:
                    return False
            return True
            
        def commit(self, x):
            self.elements.append(x)
            for e in self.elements[:-1]:
                self.diffs.add(abs(x - e))
                
        def size(self):
            return len(self.elements)
        
        def clear(self):
            self.elements = []
            self.diffs = set()

    best_score = 0
    best_A = []
    best_B = []
    
    # Pre-allocate states to avoid repeated object creation overhead if needed, 
    # but simple creation is fine for this scale.
    
    while time.time() - start_time < time_limit:
        A = SetState()
        B = SetState()
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Heuristic: Try to add to the smaller set to balance the "load" of differences
            # This helps avoid one set consuming all small differences which are more "popular"
            if A.size() < B.size():
                order = [A, B]
            elif B.size() < A.size():
                order = [B, A]
            else:
                # Random tie-break
                order = [A, B] if random.random() < 0.5 else [B, A]
            
            added = False
            for target in order:
                other = B if target is A else A
                if target.try_add(x, other.diffs):
                    target.commit(x)
                    added = True
                    break
            
            # Early termination if we hit the theoretical max
            if A.size() + B.size() == max_possible:
                break
                
        score = A.size() + B.size()
        if score > best_score:
            best_score = score
            best_A = list(A.elements)
            best_B = list(B.elements)
            
            # If we found the max, we can stop searching entirely
            if best_score == max_possible:
                break

    # Fallbacks
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
