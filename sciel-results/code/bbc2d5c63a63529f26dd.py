import math
import time
from collections import defaultdict
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
        
    # L is 30030
    
    # Generate all multiples in [0, L)
    # events: list of (value, divisor_index)
    events = []
    for i, d in enumerate(divisors):
        for m in range(0, L, d):
            events.append((m, i))
            
    # Sort by value
    events.sort(key=lambda x: x[0])
    
    # Sliding window to find min span covering all n divisors
    # We want min (events[j][0] - events[i][0]) such that window [i, j] contains all indices 0..n-1
    min_span = L
    # To handle circular nature, we can just check windows in the linear list [0, L).
    # Since the pattern repeats, any optimal window is contained in [0, L) or wraps around.
    # Wrapping around is equivalent to a window in [0, L) shifted.
    # Actually, if a window wraps, e.g. [L-10, 5], it contains values near L and near 0.
    # In the linear list [0, L), these are far apart.
    # But we can just duplicate the events: events + [(v+L, i) for v, i in events]
    # And search for windows in this extended list.
    
    extended_events = events + [(v + L, i) for v, i in events]
    N_ext = len(extended_events)
    
    counts = [0] * n
    unique_count = 0
    left = 0
    best_span = float('inf')
    
    # Find min span for covering
    for right in range(N_ext):
        val_r, idx_r = extended_events[right]
        counts[idx_r] += 1
        if counts[idx_r] == 1:
            unique_count += 1
            
        while unique_count == n:
            val_l, idx_l = extended_events[left]
            span = val_r - val_l
            if span < best_span:
                best_span = span
                
            counts[idx_l] -= 1
            if counts[idx_l] == 0:
                unique_count -= 1
            left += 1
            
    # best_span is the minimum length of an interval containing at least one multiple of each divisor.
    # Now we search for a distinct assignment.
    
    # We will iterate S from best_span upwards.
    # For each S, we check if there is a window of length S allowing a distinct assignment.
    
    def find_distinct_assignment(S):
        # Sliding window of length S over extended_events
        # We need to check windows where the span is <= S.
        # Actually, we just need to find *any* window of length S that covers all and allows distinct assignment.
        
        counts = [0] * n
        unique_count = 0
        left = 0
        
        # We store the values for each divisor in the current window
        # To make it fast, we can just re-scan or use a queue.
        # Given S is small, the number of events in a window of length S is small.
        
        for right in range(N_ext):
            val_r, idx_r = extended_events[right]
            counts[idx_r] += 1
            if counts[idx_r] == 1:
                unique_count += 1
                
            # Shrink left until window length > S or we can't shrink anymore
            while val_r - extended_events[left][0] > S:
                val_l, idx_l = extended_events[left]
                counts[idx_l] -= 1
                if counts[idx_l] == 0:
                    unique_count -= 1
                left += 1
                
            if unique_count == n:
                # Current window [left, right] has span <= S and covers all.
                # Extract values for each divisor in this window.
                # We need to iterate from left to right to get the values.
                # To avoid O(N) scan inside the loop, we can maintain a list of values?
                # But S is small, so the window size is small.
                # Let's just scan the window.
                
                vals_by_idx = defaultdict(list)
                for k in range(left, right + 1):
                    v, i = extended_events[k]
                    vals_by_idx[i].append(v)
                
                # Check if distinct assignment exists
                # Backtracking
                def can_assign(idx, used):
                    if idx == n:
                        return True
                    for v in vals_by_idx[idx]:
                        if v not in used:
                            if can_assign(idx + 1, used | {v}):
                                return True
                    return False
                
                if can_assign(0, set()):
                    # We found a valid window. Now we need to actually retrieve the values.
                    # The DFS above just checked existence.
                    # We need to return the actual values.
                    # We can modify can_assign to store the path.
                    
                    path = {}
                    def dfs(idx, used):
                        if idx == n:
                            return True
                        for v in vals_by_idx[idx]:
                            if v not in used:
                                path[idx] = v
                                if dfs(idx + 1, used | {v}):
                                    return True
                                del path[idx]
                        return False
                    
                    if dfs(0, set()):
                        return [path[i] for i in range(n)]
                        
        return None

    # Try increasing S
    # best_span is the theoretical min for covering.
    # We start from best_span.
    # Note: The span of the returned values might be less than S if the values are not at the edges.
    # But we are minimizing the span of the values.
    # If we find a valid assignment in a window of length S, the actual span is <= S.
    # We want the minimum actual span.
    # So we should iterate S from 0 upwards?
    # But we know S must be >= best_span.
    
    # Actually, if we find a valid assignment in a window of length S, the span is some value <= S.
    # We want the global minimum span.
    # So we can just iterate S from best_span to best_span + 100.
    # For each S, if we find a valid assignment, we compute the actual span.
    # We keep the best (min actual span) result.
    # Since we are iterating S upwards, the first valid assignment found might not have the min actual span?
    # No, if a window of length S allows a distinct assignment, the actual span is <= S.
    # If we find one for S=10, the span is <= 10.
    # If we find one for S=11, the span is <= 11.
    # The min possible span is at least best_span.
    # So if we find a valid assignment for S=best_span, the actual span is <= best_span.
    # But it must be >= best_span?
    # Not necessarily. The "covering" span is the min interval length containing *some* multiple for each.
    # The "distinct" span is the min interval length containing *distinct* multiples.
    # Distinct span >= Covering span.
    # So we can just return the first valid assignment found for S starting from best_span?
    # No, because for a fixed S, there might be multiple windows.
    # One window might yield span 10, another span 11 (both <= S=11).
    # We want the min span.
    # So for each S, we should find the min span among all valid windows of length S.
    # But if we find *any* valid window for S=best_span, its span is <= best_span.
    # And since Distinct Span >= Covering Span = best_span, the span must be exactly best_span?
    # Or is it possible that the distinct span is strictly greater than the covering span?
    # Yes, if the covering window has collisions.
    # In that case, we need a larger S to allow a distinct assignment.
    # So we iterate S from best_span upwards.
    # For each S, we find *any* valid distinct assignment.
    # The first one we find will have span <= S.
    # Since we start from the smallest possible S, the first valid assignment we find will have the smallest possible span?
    # Let's trace:
    # S = best_span. If valid, span <= best_span. Since span >= best_span, span = best_span.
    # If not valid, S = best_span + 1. If valid, span <= best_span + 1.
    # Could there be a solution with span = best_span that we missed?
    # No, because we checked all windows of length best_span.
    # So the first valid S gives the optimal span.
    
    for S in range(int(best_span), int(best_span) + 100):
        res = find_distinct_assignment(S)
        if res is not None:
            # Verify distinctness and divisibility
            # res is a list of 10 ints
            if len(set(res)) == n:
                for i, v in enumerate(res):
                    if v % divisors[i] != 0:
                        raise ValueError("Divisibility check failed")
                # Return
                return [int(v) for v in res]
                
    # Fallback: Should not reach here
    # Just return a valid set, maybe from a larger S
    # Or use the old method
    return divisors[:]

if __name__ == '__main__':
    res = construct()
    print(res)
