import time
import random

N = 64

# EVOLVE-BLOCK-START
class NoIsoscelesSet:
    def __init__(self):
        self.points = []
        self.point_set = set()
        # dist_sets[q] is a set of squared distances from q to other points in self.points
        self.dist_sets = {}
        
    def _dist2(self, a, b):
        return (a[0]-b[0])**2 + (a[1]-b[1])**2

    def _can_add(self, p):
        if p in self.point_set:
            return False
        k = len(self.points)
        if k == 0:
            return True
            
        dists = []
        for q in self.points:
            d = self._dist2(p, q)
            dists.append(d)
            
        # Check 1: Distances from p to all others must be distinct
        if len(set(dists)) != k:
            return False
            
        # Check 2: For each q, dist(p, q) must not be in dist_sets[q]
        for i, q in enumerate(self.points):
            d = dists[i]
            if d in self.dist_sets[q]:
                return False
        return True

    def _add(self, p):
        self.points.append(p)
        self.point_set.add(p)
        # Update dist_sets
        self.dist_sets[p] = set()
        for q in self.points[:-1]: # exclude p
            d = self._dist2(p, q)
            self.dist_sets[p].add(d)
            self.dist_sets[q].add(d)

    def _remove(self, p):
        self.points.remove(p)
        self.point_set.remove(p)
        # Update dist_sets
        dp = self.dist_sets[p]
        for q in self.points:
            d = self._dist2(p, q)
            self.dist_sets[q].remove(d)
        del self.dist_sets[p]

    def _try_insert_all(self, candidates):
        # Try to insert any candidate from the list
        # Order doesn't matter much if we just want to fill up, but random is good
        added = False
        for p in candidates:
            if p not in self.point_set:
                if self._can_add(p):
                    self._add(p)
                    added = True
        return added

def construct():
    deadline = time.time() + 280.0
    rng = random.Random(42)
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_set = None
    best_size = 0
    
    # Initial Greedy
    current = NoIsoscelesSet()
    order = all_points[:]
    rng.shuffle(order)
    for p in order:
        if current._can_add(p):
            current._add(p)
            
    best_size = len(current.points)
    best_set = current.points[:]
    
    # Local Search
    while time.time() < deadline:
        # Try to improve current state
        # 1. Try inserting any remaining points (in random order)
        unselected = [p for p in all_points if p not in current.point_set]
        rng.shuffle(unselected)
        for p in unselected:
            if current._can_add(p):
                current._add(p)
        
        if len(current.points) > best_size:
            best_size = len(current.points)
            best_set = current.points[:]
            
        # 2. Perturbation: Remove a few points to escape local optima
        # Only perturb if we are stuck (no improvements in last few tries? 
        # Or just periodically)
        # Let's do a simple swap-based search instead of blind perturbation
        # Pick a random point to remove
        if current.points:
            idx = rng.randint(0, len(current.points) - 1)
            p_remove = current.points[idx]
            current._remove(p_remove)
            
            # Try to add back p_remove or something else
            # Try to add p_remove first
            if current._can_add(p_remove):
                current._add(p_remove)
            else:
                # Try to add a random unselected point
                unselected = [p for p in all_points if p not in current.point_set]
                if unselected:
                    # Try a few random ones
                    for _ in range(10):
                        if not unselected: break
                        p_add = unselected[rng.randint(0, len(unselected)-1)]
                        if current._can_add(p_add):
                            current._add(p_add)
                            unselected.remove(p_add)
                            break
                    # If we couldn't add anything, put p_remove back
                    if p_remove not in current.point_set:
                        if current._can_add(p_remove):
                            current._add(p_remove)
                        else:
                            # This shouldn't happen if we didn't change state, 
                            # but if we removed other points? No, we only removed p_remove.
                            # So p_remove should still be valid relative to the rest.
                            # Wait, if we removed p_remove, the set is smaller.
                            # p_remove was valid in the larger set, so it is valid in the smaller set.
                            # So _can_add(p_remove) MUST be True.
                            current._add(p_remove)
                            
        # Update best
        if len(current.points) > best_size:
            best_size = len(current.points)
            best_set = current.points[:]
            
    return best_set
# EVOLVE-BLOCK-END
