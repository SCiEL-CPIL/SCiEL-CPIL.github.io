import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    N = 5
    points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(points)
    
    # Precompute line membership: for each line (defined by pair), which points lie on it
    # Use a canonical form: (dx, dy) normalized, and a reference point
    # Better: just use frozenset of all points on each line
    
    # First, find all maximal lines (sets of collinear points with >= 2 points)
    # For each pair, find all collinear points, then deduplicate by frozenset
    
    all_lines = set()  # set of frozensets
    for i in range(n_points):
        for j in range(i + 1, n_points):
            p1 = points[i]
            p2 = points[j]
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            
            pts_on_line = set([i, j])
            for k in range(n_points):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    pts_on_line.add(k)
            
            all_lines.add(frozenset(pts_on_line))
    
    # Separate lines by size
    lines_by_size = defaultdict(list)
    for line in all_lines:
        lines_by_size[len(line)].append(line)
    
    # Lines with >= 5 points are "forbidden" if all their points are selected
    forbidden_lines = [l for l in all_lines if len(l) >= 5]
    # Lines with exactly 4 points are "target" lines we want to maximize
    target_lines = [l for l in all_lines if len(l) == 4]
    # Lines with > 4 points: if we pick exactly 4 from them, they also count
    longer_lines = [l for l in all_lines if len(l) > 4]
    
    # For backtracking: we need to track, for each forbidden line, how many of its points are selected
    # If it reaches 5, that's invalid
    
    def backtrack(selected_indices, depth, best_result, best_score):
        """
        selected_indices: set of indices already selected
        depth: number of more points to select
        best_result: [best_set_of_indices, best_score]
        best_score: current best score found
        """
        if time.time() > deadline:
            return
        
        if depth == 0:
            # Compute score
            score = 0
            for line in target_lines:
                if line.issubset(frozenset(selected_indices)):
                    score += 1
            for line in longer_lines:
                if len(line & frozenset(selected_indices)) == 4:
                    score += 1
            
            if score > best_result[1]:
                best_result[0] = set(selected_indices)
                best_result[1] = score
            return
        
        # Determine which lines are "close" to being violated
        # For each forbidden line, count how many of its points are in selected_indices
        # If any forbidden line has (len(line) - 1) points selected, then we cannot select the remaining one
        
        # Find candidate points to add
        candidates = [i for i in range(n_points) if i not in selected_indices]
        
        # Prune: for each candidate, check if adding it would violate any forbidden line
        valid_candidates = []
        for c in candidates:
            # Check each forbidden line
            violated = False
            for fl in forbidden_lines:
                if c in fl:
                    # How many points of fl are already selected?
                    count = sum(1 for p in fl if p in selected_indices)
                    if count + 1 >= 5:
                        violated = True
                        break
            if not violated:
                valid_candidates.append(c)
        
        if not valid_candidates:
            return
        
        # Also prune based on whether we can still complete
        # We need 'depth' more points from valid_candidates
        if len(valid_candidates) < depth:
            return
        
        # Additional pruning: check if there are enough points that don't create issues
        # For efficiency, let's also check: for each remaining forbidden line,
        # how many more points can we select from it?
        
        # Order candidates by some heuristic? Try all for now, but limit branching
        # To keep it tractable, let's try a greedy approach first, then backtracking
        
        # Actually, let's do a simpler approach: iterative deepening with pruning
        # But given time constraints, let's try a different strategy:
        
        # Use the fact that we're on a 5x5 grid. Let's enumerate all 16-point subsets
        # that are valid (no 5 collinear) and compute their scores.
        # C(25,16) = C(25,9) = 2,042,975. That's manageable if we can prune early.
        
        # Actually, let's just do a smart search:
        # Start with a random valid set, then do local search.
        # But the directive says to use backtracking with pruning.
        
        # Let's try: pick 'depth' points from valid_candidates, with pruning at each step.
        # To limit branching, we'll use a beam-like approach: keep only top-K partial solutions.
        
        # For now, let's do a simple recursive backtracking with a depth limit.
        # If depth > 3, switch to random sampling for the remaining points.
        
        if depth <= 2:
            # Enumerate all combinations of 'depth' from valid_candidates
            for combo in itertools.combinations(valid_candidates, depth):
                new_selected = selected_indices | set(combo)
                # Check validity
                valid = True
                for fl in forbidden_lines:
                    count = sum(1 for p in fl if p in new_selected)
                    if count >= 5:
                        valid = False
                        break
                if valid:
                    # Compute partial score contribution
                    # Actually, just recurse
                    backtrack(new_selected, 0, best_result, best_result[1])
        else:
            # For larger depth, use a heuristic: pick the most promising candidates
            # Score each candidate by how many target lines it completes
            def candidate_score(c):
                s = 0
                for tl in target_lines:
                    if c in tl:
                        # How many other points of tl are already selected?
                        other_count = sum(1 for p in tl if p in selected_indices and p != c)
                        if other_count == 3:
                            s += 10  # Completing a target line
                        elif other_count == 2:
                            s += 1
                for ll in longer_lines:
                    if c in ll:
                        other_count = sum(1 for p in ll if p in selected_indices and p != c)
                        if other_count == 3:
                            s += 5  # Making a 4-subset of a longer line
                        elif other_count == 2:
                            s += 1
                return s
            
            # Sort candidates by score, descending
            valid_candidates.sort(key=candidate_score, reverse=True)
            
            # Take top min(len(valid_candidates), depth*3) candidates
            top_k = min(len(valid_candidates), depth * 3)
            top_candidates = valid_candidates[:top_k]
            
            for c in top_candidates:
                new_selected = selected_indices | {c}
                backtrack(new_selected, depth - 1, best_result, best_result[1])
    
    # Initialize best result
    best_result = [set(), 0]
    
    # Start backtracking: we need to select 16 points total
    # Start with empty set
    backtrack(set(), 16, best_result, 0)
    
    # If backtracking found a good result, use it
    if best_result[0]:
        best_indices = best_result[0]
    else:
        # Fallback: random search with local improvement
        best_indices = set()
        best_score = 0
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Random initial set
            current = set(random.sample(range(n_points), 16))
            
            # Check validity
            valid = True
            for fl in forbidden_lines:
                count = sum(1 for p in fl if p in current)
                if count >= 5:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute score
            score = 0
            for tl in target_lines:
                if tl.issubset(current):
                    score += 1
            for ll in longer_lines:
                if len(ll & current) == 4:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_indices = set(current)
            
            # Local search
            for _ in range(1000):
                if time.time() > deadline:
                    break
                
                p_in = random.choice(list(current))
                p_out = random.choice([p for p in range(n_points) if p not in current])
                
                new_current = set(current)
                new_current.remove(p_in)
                new_current.add(p_out)
                
                # Check validity
                valid = True
                for fl in forbidden_lines:
                    count = sum(1 for p in fl if p in new_current)
                    if count >= 5:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Compute score
                new_score = 0
                for tl in target_lines:
                    if tl.issubset(new_current):
                        new_score += 1
                for ll in longer_lines:
                    if len(ll & new_current) == 4:
                        new_score += 1
                
                if new_score >= score:
                    score = new_score
                    current = new_current
                    if score > best_score:
                        best_score = score
                        best_indices = set(current)
    
    # Convert to list of lists
    return [[points[i][0], points[i][1]] for i in sorted(best_indices)]
