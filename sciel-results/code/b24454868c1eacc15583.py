import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 82s limit, 8s buffer
    
    n = 256
    
    # Helper: check if adding x to a set creates a 3-AP
    # Returns True if x can be added without creating a 3-AP
    def can_add(x, selected_set):
        for a in selected_set:
            # Check if x, a, b form AP for some b in selected_set
            # Case 1: b = 2*a - x (x, a, b with a as midpoint)
            b1 = 2 * a - x
            if b1 in selected_set and b1 != x and b1 != a:
                return False
            # Case 2: a, x, b with x as midpoint: b = 2*x - a
            b2 = 2 * x - a
            if b2 in selected_set and b2 != x and b2 != a:
                return False
            # Case 3: a, b, x with b as midpoint: b = (a+x)/2
            if (a + x) % 2 == 0:
                b3 = (a + x) // 2
                if b3 in selected_set and b3 != x and b3 != a:
                    return False
        return True
    
    # Helper: calculate the reciprocal sum
    def calc_score(selected):
        return sum(Fraction(1, x) for x in selected)
    
    # Helper: full validation
    def is_valid(selected_set):
        lst = list(selected_set)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                # Check if there's a c such that a,b,c form AP
                # c = 2*b - a
                c1 = 2 * b - a
                if c1 in selected_set and c1 != a and c1 != b:
                    return False
                # c = 2*a - b
                c2 = 2 * a - b
                if c2 in selected_set and c2 != a and c2 != b:
                    return False
                # Check if a, b, c with c as midpoint
                if (a + b) % 2 == 0:
                    c3 = (a + b) // 2
                    if c3 in selected_set and c3 != a and c3 != b:
                        return False
        return True
    
    # Initial greedy construction
    selected_set = set()
    for x in range(1, n + 1):
        if can_add(x, selected_set):
            selected_set.add(x)
    
    # Local search with multi-swap
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try single swaps first
        candidates = sorted([v for v in range(1, n + 1) if v not in selected_set])
        
        for v in candidates:
            if time.time() > deadline:
                break
            
            # Find all u in selected_set such that removing u allows adding v
            # An element a in selected_set conflicts with v if there exists b in selected_set
            # such that {a, b, v} is a 3-AP.
            
            # Collect all elements that are part of any AP with v
            conflict_elements = set()
            for a in selected_set:
                # Check if v forms AP with a and some b in selected_set
                # Case 1: b = 2*a - v
                b1 = 2 * a - v
                if b1 in selected_set and b1 != v and b1 != a:
                    conflict_elements.add(a)
                    conflict_elements.add(b1)
                
                # Case 2: b = (a+v)/2
                if (a + v) % 2 == 0:
                    b2 = (a + v) // 2
                    if b2 in selected_set and b2 != v and b2 != a:
                        conflict_elements.add(a)
                        conflict_elements.add(b2)
                
                # Case 3: b = 2*v - a
                b3 = 2 * v - a
                if b3 in selected_set and b3 != v and b3 != a:
                    conflict_elements.add(a)
                    conflict_elements.add(b3)
            
            if not conflict_elements:
                # v can be added directly
                selected_set.add(v)
                improved = True
                break
            
            # Try single removal: find u such that removing u eliminates all conflicts
            for u in conflict_elements:
                if u not in selected_set:
                    continue
                if v >= u:  # No gain
                    continue
                
                # Check if removing u makes v valid
                temp_set = selected_set - {u}
                if can_add(v, temp_set):
                    selected_set.remove(u)
                    selected_set.add(v)
                    improved = True
                    break
            
            if improved:
                break
        
        # If no single swap worked, try double swaps
        if not improved and time.time() < deadline:
            # Try removing two elements and adding one
            candidates = sorted([v for v in range(1, n + 1) if v not in selected_set])
            
            for v in candidates[:20]:  # Limit to top 20 candidates for speed
                if time.time() > deadline:
                    break
                
                # Find conflict elements
                conflict_elements = set()
                for a in selected_set:
                    b1 = 2 * a - v
                    if b1 in selected_set and b1 != v and b1 != a:
                        conflict_elements.add(a)
                        conflict_elements.add(b1)
                    
                    if (a + v) % 2 == 0:
                        b2 = (a + v) // 2
                        if b2 in selected_set and b2 != v and b2 != a:
                            conflict_elements.add(a)
                            conflict_elements.add(b2)
                    
                    b3 = 2 * v - a
                    if b3 in selected_set and b3 != v and b3 != a:
                        conflict_elements.add(a)
                        conflict_elements.add(b3)
                
                if len(conflict_elements) <= 2:
                    # Try removing all conflict elements
                    if len(conflict_elements) == 0:
                        selected_set.add(v)
                        improved = True
                        break
                    
                    # Try removing pairs from conflict_elements
                    conf_list = list(conflict_elements)
                    for i in range(len(conf_list)):
                        for j in range(i + 1, len(conf_list)):
                            u1, u2 = conf_list[i], conf_list[j]
                            if v >= u1 and v >= u2:
                                continue
                            temp_set = selected_set - {u1, u2}
                            if can_add(v, temp_set):
                                selected_set.remove(u1)
                                selected_set.remove(u2)
                                selected_set.add(v)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Also try adding multiple small elements
            if not improved:
                small_candidates = sorted([v for v in range(1, 30) if v not in selected_set])
                for v in small_candidates:
                    if time.time() > deadline:
                        break
                    temp_set = set(selected_set)
                    if can_add(v, temp_set):
                        selected_set.add(v)
                        improved = True
                        break
    
    # Final validation
    if not is_valid(selected_set):
        # Fallback: rebuild
        selected_set = set()
        for x in range(1, n + 1):
            if can_add(x, selected_set):
                selected_set.add(x)
    
    return list(selected_set)

# EVOLVE-BLOCK-END
