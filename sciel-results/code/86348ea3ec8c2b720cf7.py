import time
from itertools import combinations

def _is_valid(selected, candidate):
    """Check if adding candidate to selected maintains validity."""
    # Check candidate doesn't divide sum of any 3 distinct other elements in selected
    others = selected
    for combo in combinations(others, 3):
        s = combo[0] + combo[1] + combo[2]
        if candidate != 0 and s % candidate == 0:
            return False
    # Check each existing element doesn't divide sum of 3 distinct others including candidate
    for a in selected:
        # Need 3 distinct others from (selected - {a}) ∪ {candidate}
        # Cases: 3 from selected (already checked), 2 from selected + candidate, 1 from selected + candidate (impossible since need 3)
        # So check: 2 from selected (not a) + candidate
        for combo in combinations([x for x in selected if x != a], 2):
            s = combo[0] + combo[1] + candidate
            if s % a == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = 55  # seconds, safety margin before 60s
    
    # Greedy: try numbers from large to small, but with smarter ordering
    # Numbers that are harder to divide sums should go first
    # Large numbers are safer
    
    selected = []
    # Try all numbers from 192 down to 1
    for x in range(192, 0, -1):
        if time.time() - start_time > deadline:
            break
        if _is_valid(selected, x):
            selected.append(x)
    
    # Try to improve with local search: swap or add
    # But first, let's see what we have
    # The greedy from large to small should give a good set
    
    # Try adding numbers we might have skipped by removing others
    # Simple local search: try to add any number from 1..192 not in selected
    # by removing one or two existing elements
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        current_size = len(selected)
        sel_set = set(selected)
        
        # Try to add each missing number
        for x in range(192, 0, -1):
            if x in sel_set:
                continue
            if time.time() - start_time > deadline:
                break
            
            # Try adding x directly
            if _is_valid(selected, x):
                selected.append(x)
                sel_set.add(x)
                improved = True
                continue
            
            # Try removing one element to add x
            for i, a in enumerate(selected):
                new_sel = selected[:i] + selected[i+1:]
                if _is_valid(new_sel, x):
                    selected = new_sel + [x]
                    sel_set.remove(a)
                    sel_set.add(x)
                    improved = True
                    break
            else:
                continue
            break
    
    return selected
