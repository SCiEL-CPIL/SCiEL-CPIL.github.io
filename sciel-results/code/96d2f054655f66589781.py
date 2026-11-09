import time
import random
import heapq

# Precomputation
def precompute_aps(n):
    aps = []
    for d in range(1, n//3 + 1):
        for a in range(1, n - 3*d + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= n:
                aps.append((a, b, c, e))
    return aps

APS = precompute_aps(512)
APS_CONTAINING = [[] for _ in range(513)]
for i, ap in enumerate(APS):
    for x in ap:
        APS_CONTAINING[x].append(i)

def get_base3_set(n):
    s = set()
    for x in range(1, n+1):
        # check base 3 digits
        y = x
        valid = True
        while y > 0:
            if y % 3 == 2:
                valid = False
                break
            y //= 3
        if valid:
            s.add(x)
    return s

def greedy_fill_smart(initial_set, n=512):
    S = set(initial_set)
    ap_counts = [0] * len(APS)
    
    # Initialize counts
    for x in S:
        for ap_idx in APS_CONTAINING[x]:
            ap_counts[ap_idx] += 1
            
    # Candidates not in S
    candidates = [x for x in range(1, n+1) if x not in S]
    
    # We will use a heap to select the "best" candidate next.
    # Score: number of APs that would go from 2 to 3 (creating a blocked AP for others).
    # Lower is better.
    # Tie-breaker: random.
    
    # To avoid O(N log N) re-heapification on every add, we can just iterate and pick min, 
    # but N=512 is small. Let's just re-evaluate scores or use a simple list scan if heap maintenance is complex.
    # Given N=512, a simple scan for min score is O(N * AvgAPs) ~ 512 * 20 = 10k ops. 
    # 512 additions * 10k = 5M ops. This is fast enough in Python? Maybe.
    # Let's try a simpler approach: Shuffle and sort by a heuristic score calculated once? No, scores change.
    
    # Let's stick to a randomized greedy but with a "look-ahead" tie breaker or just multiple random passes.
    # Actually, let's try the heap approach properly.
    
    # Since scores change, we'll just do a loop:
    # 1. Find all valid candidates.
    # 2. Among valid candidates, pick the one with lowest "conflict potential".
    # 3. Add it.
    # 4. Repeat.
    
    remaining = set(candidates)
    
    while remaining:
        best_x = -1
        best_score = float('inf')
        
        # Sample a subset if remaining is large to speed up? 
        # For 512, full scan is fine.
        for x in remaining:
            # Check if valid
            valid = True
            score = 0
            for ap_idx in APS_CONTAINING[x]:
                if ap_counts[ap_idx] >= 3:
                    valid = False
                    break
                elif ap_counts[ap_idx] == 2:
                    score += 1
            if valid:
                # Add small random noise to break ties and allow exploration
                score += random.random() * 0.1
                if score < best_score:
                    best_score = score
                    best_x = x
        
        if best_x == -1:
            break
            
        # Add best_x
        S.add(best_x)
        remaining.remove(best_x)
        for ap_idx in APS_CONTAINING[best_x]:
            ap_counts[ap_idx] += 1
            
    return S

def construct():
    start_time = time.time()
    best_S = get_base3_set(512)
    best_size = len(best_S)
    
    # Run local search
    while time.time() - start_time < 50:
        # Perturb
        current_S = set(best_S)
        
        # Try different removal strategies
        # 1. Random removal
        # 2. Remove elements with high "degree" (part of many APs)
        
        if random.random() < 0.5:
            # Random removal
            removal_rate = random.choice([0.02, 0.05, 0.1])
            num_remove = max(1, int(len(current_S) * removal_rate))
            to_remove = random.sample(list(current_S), num_remove)
            current_S -= set(to_remove)
        else:
            # Remove "central" elements (heuristic)
            # Elements in many APs are more "expensive" to keep? 
            # Actually, removing elements that are in many *blocked* APs might help.
            # For now, just random small kick.
            num_remove = random.randint(1, 3)
            to_remove = random.sample(list(current_S), num_remove)
            current_S -= set(to_remove)
        
        new_S = greedy_fill_smart(current_S)
        
        if len(new_S) > best_size:
            best_size = len(new_S)
            best_S = new_S
            
    # Final polish: try adding in sorted order
    final_S = set(best_S)
    ap_counts = [0] * len(APS)
    for x in final_S:
        for ap_idx in APS_CONTAINING[x]:
            ap_counts[ap_idx] += 1
            
    for x in range(1, 513):
        if x not in final_S:
            valid = True
            for ap_idx in APS_CONTAINING[x]:
                if ap_counts[ap_idx] >= 3:
                    valid = False
                    break
            if valid:
                final_S.add(x)
                for ap_idx in APS_CONTAINING[x]:
                    ap_counts[ap_idx] += 1
    # And reverse
    ap_counts = [0] * len(APS)
    for x in final_S:
        for ap_idx in APS_CONTAINING[x]:
            ap_counts[ap_idx] += 1
    for x in range(512, 0, -1):
        if x not in final_S:
            valid = True
            for ap_idx in APS_CONTAINING[x]:
                if ap_counts[ap_idx] >= 3:
                    valid = False
                    break
            if valid:
                final_S.add(x)
                for ap_idx in APS_CONTAINING[x]:
                    ap_counts[ap_idx] += 1
                    
    if len(final_S) > best_size:
        best_S = final_S
        best_size = len(final_S)

    return sorted(best_S)
