import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0

    def are_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1])

    def is_valid_move(points, idx, new_p):
        """Check if replacing points[idx] with new_p keeps the set valid (no 4 collinear)."""
        # The new set is S' = (S \ {points[idx]}) U {new_p}
        # A 4-collinearity can only involve new_p and 3 other points from S \ {points[idx]}
        # Because any 4-collinearity not involving new_p was already valid in S (since we only removed a point).
        # Wait, if S had a 4-collinearity involving points[idx], removing it breaks it.
        # If S did not have a 4-collinearity, removing a point doesn't create one.
        # So we only need to check if new_p forms a 4-collinear set with any 3 points from S \ {points[idx]}.
        
        others = [p for i, p in enumerate(points) if i != idx]
        
        # Check all triples from 'others'
        n = len(others)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(others[i], others[j], others[k]):
                        # If these 3 are collinear, check if new_p is on the same line
                        if are_collinear(others[i], others[j], new_p):
                            return False
        return True

    def count_triples(points):
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def generate_valid_set():
        points = []
        while len(points) < 20:
            p = random_point()
            if p in points:
                continue
            # Check if adding p creates 4 collinear
            # We can use similar logic: check if p is collinear with any triple in points
            # But points size is < 20, so O(n^3) is fine.
            valid = True
            n = len(points)
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if are_collinear(points[i], points[j], points[k]):
                            if are_collinear(points[i], points[j], p):
                                valid = False
                                break
                    if not valid: break
                if not valid: break
            if valid:
                points.append(p)
        return points

    best_points = None
    best_triples = -1
    
    current_points = generate_valid_set()
    current_triples = count_triples(current_points)
    best_points = current_points
    best_triples = current_triples
    
    temp = 10.0
    cooling = 0.9995
    
    while time.time() - start_time < time_limit:
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        new_p = random_point()
        
        if new_p == old_p:
            continue
        
        # Check distinctness with others
        # is_valid_move doesn't check distinctness with others, but collinearity with 3 points
        # If new_p is same as another point, are_collinear might behave weirdly?
        # If p1 == p2, slope is 0/0.
        # Let's explicitly check distinctness.
        is_distinct = True
        for i, p in enumerate(current_points):
            if i != idx and p == new_p:
                is_distinct = False
                break
        if not is_distinct:
            continue

        if not is_valid_move(current_points, idx, new_p):
            continue
            
        new_triples = count_triples(current_points) # Wait, this is O(N^3) = 1140 ops.
        # We can calculate delta more efficiently?
        # Removing old_p removes triples involving old_p.
        # Adding new_p adds triples involving new_p.
        # Triples not involving old_p or new_p remain same.
        
        # Triples involving old_p:
        old_triples = 0
        others = [p for i, p in enumerate(current_points) if i != idx]
        for i in range(len(others)):
            for j in range(i+1, len(others)):
                if are_collinear(old_p, others[i], others[j]):
                    old_triples += 1
        
        new_triples_add = 0
        for i in range(len(others)):
            for j in range(i+1, len(others)):
                if are_collinear(new_p, others[i], others[j]):
                    new_triples_add += 1
        
        delta = new_triples_add - old_triples
        new_total = current_triples + delta
        
        if delta > 0 or random.random() < pow(2.718, delta / temp):
            current_points[idx] = new_p
            current_triples = new_total
            
            if current_triples > best_triples:
                best_triples = current_triples
                best_points = current_points
                
        temp *= cooling
        
    return best_points
