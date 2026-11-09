import time
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy
    
    def can_add(pts, dist_sets, p):
        """Check if point p can be added without creating isosceles triangle."""
        k = len(pts)
        if k == 0:
            return True
        # Compute squared distances from p to all existing points
        dists = [sq_dist(p, pts[i]) for i in range(k)]
        # Check for duplicate distances (isosceles with apex at p)
        if len(set(dists)) < k:
            return False
        # Check if any distance from p to pts[i] matches an existing distance in dist_sets[pts[i]]
        for i in range(k):
            if dists[i] in dist_sets[pts[i]]:
                return False
        return True
    
    def try_add(pts, dist_sets, p):
        """Add point p to pts and update dist_sets. Returns True if successful."""
        k = len(pts)
        for i in range(k):
            d = sq_dist(p, pts[i])
            dist_sets[pts[i]].add(d)
        dist_sets[p] = set(d for d in [sq_dist(p, pts[i]) for i in range(k)])
        pts.append(p)
        return True
    
    def greedy_construct(order):
        """Greedy construction in given order."""
        pts = []
        dist_sets = {}
        for p in order:
            if can_add(pts, dist_sets, p):
                try_add(pts, dist_sets, p)
        return pts, dist_sets
    
    def local_search(pts, dist_sets, max_iters=50000):
        """Improve solution by trying to add points and remove/re-add."""
        pts = list(pts)
        dist_sets = {p: set(ds) for p, ds in dist_sets.items()}
        
        all_points = [(x, y) for x in range(N) for y in range(N)]
        pt_set = set(pts)
        
        # Try adding points
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Shuffle candidate order for diversity
            candidates = [p for p in all_points if p not in pt_set]
            random.shuffle(candidates)
            
            for p in candidates:
                if can_add(pts, dist_sets, p):
                    try_add(pts, dist_sets, p)
                    pt_set.add(p)
                    improved = True
                    break
            if not improved:
                break
        
        # Try swap operations: remove one point, try to add two
        for _ in range(max_iters // 2):
            if time.time() > deadline:
                break
            improved = False
            if len(pts) < 2:
                break
            idx = random.randint(0, len(pts) - 1)
            removed = pts[idx]
            
            # Remove from data structures
            for i, p in enumerate(pts):
                if i == idx:
                    continue
                d = sq_dist(removed, p)
                dist_sets[p].discard(d)
            pts.pop(idx)
            del dist_sets[removed]
            pt_set.discard(removed)
            
            # Try to add two points
            added = []
            candidates = [p for p in all_points if p not in pt_set]
            random.shuffle(candidates)
            for p in candidates:
                if can_add(pts, dist_sets, p):
                    try_add(pts, dist_sets, p)
                    pt_set.add(p)
                    added.append(p)
                    break
            for p in candidates:
                if p in pt_set and p not in added:
                    continue
                if p in added:
                    continue
                if can_add(pts, dist_sets, p):
                    try_add(pts, dist_sets, p)
                    pt_set.add(p)
                    added.append(p)
                    break
            
            if len(added) >= 2:
                improved = True
            else:
                # Revert
                for p in added:
                    for i, q in enumerate(pts):
                        if i == pts.index(p):
                            continue
                        d = sq_dist(p, q)
                        dist_sets[q].discard(d)
                    pts.remove(p)
                    del dist_sets[p]
                    pt_set.discard(p)
                # Re-add removed point
                for p in pts:
                    d = sq_dist(removed, p)
                    dist_sets[p].add(d)
                dist_sets[removed] = set(sq_dist(removed, p) for p in pts)
                pts.insert(idx, removed)
                pt_set.add(removed)
            
            if not improved:
                break
        
        return pts, dist_sets
    
    best_pts = []
    best_size = 0
    
    # Multiple restarts with different random orderings
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Create random ordering of all grid points
        all_points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(all_points)
        
        # Greedy construction
        pts, dist_sets = greedy_construct(all_points)
        
        if len(pts) > best_size:
            best_size = len(pts)
            best_pts = list(pts)
        
        # Local search to improve
        pts, dist_sets = local_search(pts, dist_sets, max_iters=20000)
        
        if len(pts) > best_size:
            best_size = len(pts)
            best_pts = list(pts)
        
        # Early exit if we've used too much time
        if time.time() > deadline - 30:
            break
    
    return best_pts
# EVOLVE-BLOCK-END
