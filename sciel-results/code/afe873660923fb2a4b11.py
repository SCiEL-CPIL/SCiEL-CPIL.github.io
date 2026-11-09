"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from fractions import Fraction

def construct():
    """
    Find the largest subset of [1, 24] such that no reciprocal of a selected element
    equals the sum of reciprocals of a subset of the OTHER selected elements.
    
    Uses a backtracking search with forward checking. We process elements in a specific order
    and maintain the set of 'forbidden sums' that would conflict with already-selected elements.
    """
    N = 24
    elements = list(range(1, N + 1))
    
    # Precompute reciprocals as Fractions for exact arithmetic
    recips = {x: Fraction(1, x) for x in elements}
    
    # We want to maximize cardinality.
    # Strategy: Use a backtracking search with forward checking.
    # We process elements in decreasing order (24, 23, ..., 1).
    # This order is beneficial because larger elements have smaller reciprocals,
    # making it less likely that their reciprocals are sums of other reciprocals.
    # Also, when we add a new element x (smaller than existing ones),
    # the check for existing elements y (larger than x) involves checking if 1/y - 1/x
    # is a sum of a subset of the other existing elements. Since 1/y < 1/x is false
    # (y > x => 1/y < 1/x), wait.
    # If we process in decreasing order, existing elements are LARGER than x.
    # So for an existing element y (y > x), we check if 1/y is a sum of a subset of (S\{y}) U {x}.
    # This is equivalent to checking if 1/y - 1/x is a sum of a subset of S\{y}.
    # But 1/y - 1/x is negative since y > x.
    # Sums of reciprocals are positive.
    # So 1/y - 1/x < 0 can never be a sum of positive reciprocals.
    # Therefore, adding a smaller element x NEVER breaks the validity of larger existing elements!
    # This is a huge simplification.
    
    # Let's verify this logic:
    # Condition for y to be valid in S' = S U {x}:
    # 1/y is not in Sums(S' \ {y}).
    # S' \ {y} = (S \ {y}) U {x}.
    # Sums((S\{y}) U {x}) = Sums(S\{y}) U {1/x} U {s + 1/x for s in Sums(S\{y})}.
    # We know 1/y is not in Sums(S\{y}) (by induction, S is valid).
    # We need to check if 1/y is in {1/x} U {s + 1/x for s in Sums(S\{y})}.
    # Case 1: 1/y == 1/x => y == x. Impossible since x is new and distinct.
    # Case 2: 1/y == s + 1/x => 1/y - 1/x == s.
    # Since y > x (processing in decreasing order), 1/y < 1/x, so 1/y - 1/x < 0.
    # But s is a sum of positive reciprocals, so s > 0.
    # Thus, 1/y - 1/x cannot equal s.
    # So, adding x does NOT break the validity of any existing element y > x.
    
    # Therefore, the only constraint when adding x is:
    # 1. 1/x is not in Sums(S). (i.e., x is not a subsum of the existing set S)
    
    # This means we can simply build the set greedily in decreasing order!
    # Start with S = [].
    # For x in [24, 23, ..., 1]:
    #   If 1/x is not in Sums(S), add x to S.
    #   Update Sums(S).
    
    # Wait, is this optimal?
    # Greedy might not be optimal. We might skip a small x now to allow a different combination later?
    # No, the decision to include or exclude x only affects future (smaller) elements.
    # But the condition for future elements z < x is:
    # 1/z is not in Sums(S U {x} U {other smaller ones}).
    # If we include x, Sums(S U {x}) is larger than Sums(S).
    # So including x might make it harder to include future elements z (because 1/z might become a sum).
    # So we should be careful. We might want to EXCLUDE some x to allow more z's.
    
    # So greedy is not necessarily optimal. We need backtracking.
    # But the forward checking is very simple:
    # When considering x, we only need to check if 1/x is in Sums(S).
    # If it is, we MUST skip x.
    # If it is not, we can choose to include or skip x.
    # But if we include x, we update Sums(S) to Sums(S) U {1/x} U {s + 1/x for s in Sums(S)}.
    
    # This is a much simpler backtracking problem than the general one.
    # State: (idx, current_set, sums)
    # idx: next element to consider (starting from 24 down to 1)
    # current_set: list of selected elements
    # sums: set of all subset sums of reciprocals of current_set
    
    # Pruning:
    # If len(current_set) + (number of remaining elements) <= best_size, prune.
    
    # Let's implement this.
    
    best_size = 0
    best_set = []
    
    # Process elements in decreasing order
    ordered_elements = list(range(N, 0, -1))
    
    def backtrack(idx, current_set, sums):
        nonlocal best_size, best_set
        
        # Pruning: if even taking all remaining elements can't beat best_size, return
        if len(current_set) + (N - idx) <= best_size:
            return
        
        if idx == N:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        x = ordered_elements[idx]
        rx = recips[x]
        
        # Option 1: Skip x
        backtrack(idx + 1, current_set, sums)
        
        # Option 2: Take x
        # Check if rx is in sums
        if rx not in sums:
            # Tentatively take x
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + rx)
            new_sums.add(rx)
            
            current_set.append(x)
            backtrack(idx + 1, current_set, new_sums)
            current_set.pop()
            
    backtrack(0, [], set())
    
    if not best_set:
        best_set = [1]
        
    return best_set

# EVOLVE-BLOCK-END
