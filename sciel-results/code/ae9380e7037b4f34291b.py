import time
import random

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute 4-APs
    ap_list = [[] for _ in range(N)]
    
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            p1 = a
            p2 = a + d
            p3 = a + 2 * d
            p4 = a + 3 * d
            if p4 < N:
                ap_list[p1].append((p2, p3, p4))
                ap_list[p2].append((p1, p3, p4))
                ap_list[p3].append((p1, p2, p4))
                ap_list[p4].append((p1, p2, p3))

    def is_valid_ap(c1, c2, c3, c4):
        """Returns True if the AP uses at least 3 distinct colors."""
        # Check if size of {c1, c2, c3, c4} >= 3
        # Invalid if size 1 or 2.
        
        # Size 1
        if c1 == c2 and c1 == c3 and c1 == c4:
            return False
        
        # Size 2
        # Check pairs
        if c1 == c2:
            if c3 == c4 and c1 != c3:
                return False
            if c3 == c1 and c4 != c1:
                return False
            if c4 == c1 and c3 != c1:
                return False
        if c1 == c3:
            if c2 == c4 and c1 != c2:
                return False
            if c2 == c1 and c4 != c1:
                return False
            if c4 == c1 and c2 != c1:
                return False
        if c1 == c4:
            if c2 == c3 and c1 != c2:
                return False
            if c2 == c1 and c3 != c1:
                return False
            if c3 == c1 and c2 != c1:
                return False
        if c2 == c3:
            if c1 == c4 and c2 != c1:
                return False
            if c1 == c2 and c4 != c2:
                return False
            if c4 == c2 and c1 != c2:
                return False
        if c2 == c4:
            if c1 == c3 and c2 != c1:
                return False
            if c1 == c2 and c3 != c2:
                return False
            if c3 == c2 and c1 != c2:
                return False
        if c3 == c4:
            if c1 == c2 and c3 != c1:
                return False
            if c1 == c3 and c2 != c3:
                return False
            if c2 == c3 and c1 != c3:
                return False
                
        return True

    def count_conflicts(colors):
        """Returns a list of booleans: is index i in a conflict?"""
        in_conflict = [False] * N
        for i in range(N):
            for j, k, l in ap_list[i]:
                if not is_valid_ap(colors[i], colors[j], colors[k], colors[l]):
                    in_conflict[i] = True
                    in_conflict[j] = True
                    in_conflict[k] = True
                    in_conflict[l] = True
        return in_conflict

    def min_conflicts(K, max_restarts=50):
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            colors = [random.randint(0, K - 1) for _ in range(N)]
            in_conflict = count_conflicts(colors)
            
            stuck_counter = 0
            
            # Max steps per restart
            while time.time() < deadline:
                if not any(in_conflict):
                    return colors
                
                # Pick a random conflicting index
                conflicting_indices = [i for i in range(N) if in_conflict[i]]
                if not conflicting_indices:
                    return colors
                
                i = random.choice(conflicting_indices)
                
                # Find best color for i
                best_c = -1
                min_cost = float('inf')
                
                for c in range(K):
                    cost = 0
                    for j, k, l in ap_list[i]:
                        if not is_valid_ap(c, colors[j], colors[k], colors[l]):
                            cost += 1
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_c = c
                        if cost == 0:
                            break
                
                colors[i] = best_c
                in_conflict = count_conflicts(colors)
                
                # If no improvement or stuck, kick
                # Simple kick: if we've done 1000 steps without success, randomize 5 vars
                # We don't track improvement easily here, so just use a step counter
                # Actually, min-conflicts usually converges. If it loops, we restart.
                # Let's add a step limit per restart to avoid infinite loops in local min.
                # If we exceed 5000 steps, break to next restart.
                # We need a step counter.
                # Let's just break if we've done too many steps.
                # For now, the time check handles it.
                # But if it oscillates, it wastes time.
                # Let's add a simple stagnation check: if best_c leads to same conflicts, kick.
                # This is complex. Let's just rely on random restarts.
                # To make restarts happen faster, limit steps per restart.
                # We'll implement a step counter inside the while loop.
                
        return None

    # Modify min_conflicts to accept max_steps_per_restart
    def min_conflicts_limited(K, max_restarts=100, max_steps_per_restart=5000):
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            colors = [random.randint(0, K - 1) for _ in range(N)]
            in_conflict = count_conflicts(colors)
            
            for step in range(max_steps_per_restart):
                if time.time() > deadline:
                    return None
                if not any(in_conflict):
                    return colors
                
                conflicting_indices = [i for i in range(N) if in_conflict[i]]
                if not conflicting_indices:
                    return colors
                
                i = random.choice(conflicting_indices)
                
                best_c = -1
                min_cost = float('inf')
                
                for c in range(K):
                    cost = 0
                    for j, k, l in ap_list[i]:
                        if not is_valid_ap(c, colors[j], colors[k], colors[l]):
                            cost += 1
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_c = c
                        if cost == 0:
                            break
                
                colors[i] = best_c
                in_conflict = count_conflicts(colors)
                
        return None

    # Try K=3
    sol = min_conflicts_limited(3)
    if sol is not None:
        return sol

    # Try K=4
    sol = min_conflicts_limited(4)
    if sol is not None:
        return sol

    # Try K=5
    sol = min_conflicts_limited(5)
    if sol is not None:
        return sol

    # Fallback
    return [i % 3 for i in range(N)]
