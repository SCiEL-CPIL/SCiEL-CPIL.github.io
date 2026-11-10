import time
from math import isqrt, gcd
from fractions import Fraction
import random

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _is_squarefree(n):
    if n == 1:
        return True
    r = isqrt(n)
    for i in range(2, r + 1):
        if n % (i * i) == 0:
            return False
    return True

def _check_valid(points, D, max_numer=10**40):
    n = len(points)
    if n < 3 or n > 12:
        return False
    if D < 1 or D > 10**12:
        return False
    if not _is_squarefree(D):
        return False
    
    # Check distinctness
    for i in range(n):
        for j in range(i+1, n):
            if points[i] == points[j]:
                return False

    # Check pairwise distances rational
    for i in range(n):
        for j in range(i+1, n):
            dx_num = points[i][0][0]*points[j][0][1] - points[j][0][0]*points[i][0][1]
            dx_den = points[i][0][1]*points[j][0][1]
            dy_num = points[i][1][0]*points[j][1][1] - points[j][1][0]*points[i][1][1]
            dy_den = points[i][1][1]*points[j][1][1]
            
            # dist^2 = (dx)^2 + D*(dy)^2
            num = dx_num*dx_num * dy_den*dy_den + D * dy_num*dy_num * dx_den*dx_den
            den = dx_den*dx_den * dy_den*dy_den
            if den == 0:
                return False
            if not _is_square(num * den):
                return False

    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1n, x1d = points[i][0]
                y1n, y1d = points[i][1]
                x2n, x2d = points[j][0]
                y2n, y2d = points[j][1]
                x3n, x3d = points[k][0]
                y3n, y3d = points[k][1]
                
                def sub_frac(n1, d1, n2, d2):
                    return n1*d2 - n2*d1, d1*d2
                
                dx12_n, dx12_d = sub_frac(x2n, x2d, x1n, x1d)
                dy12_n, dy12_d = sub_frac(y2n, y2d, y1n, y1d)
                dx13_n, dx13_d = sub_frac(x3n, x3d, x1n, x1d)
                dy13_n, dy13_d = sub_frac(y3n, y3d, y1n, y1d)
                
                num = dx12_n*dy13_n*dy12_d*dx13_d - dy12_n*dx13_n*dx12_d*dy13_d
                if num == 0:
                    return False

    # Check no four concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    def get_row(pt):
                        x = Fraction(pt[0][0], pt[0][1])
                        y = Fraction(pt[1][0], pt[1][1])
                        return [x*x + D*y*y, x, y, 1]
                    
                    M = [
                        get_row(points[i]),
                        get_row(points[j]),
                        get_row(points[k]),
                        get_row(points[l])
                    ]
                    
                    def det3(m):
                        return (m[0][0]*(m[1][1]*m[2][2]-m[2][1]*m[1][2]) -
                                m[0][1]*(m[1][0]*m[2][2]-m[2][0]*m[1][2]) +
                                m[0][2]*(m[1][0]*m[2][1]-m[2][0]*m[1][1]))
                    
                    det = 0
                    for c in range(4):
                        minor = [row[:c] + row[c+1:] for row in M[1:]]
                        det += ((-1)**c) * M[0][c] * det3(minor)
                    
                    if det == 0:
                        return False

    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We will search for integer points with D=1
    # Start with a known good set
    base_points = [(0, 0), (3, 4), (5, 12), (16, 30), (21, 28)]
    best_points = base_points
    best_size = len(best_points)
    
    # Generate a pool of candidate points
    R = 60
    candidates = []
    for x in range(-R, R+1):
        for y in range(-R, R+1):
            if (x, y) not in base_points:
                candidates.append((x, y))
    
    random.shuffle(candidates)
    
    # Precompute distances between base points and candidates
    # We'll use a greedy approach to add points
    
    current_set = list(base_points)
    
    # Sort candidates by some heuristic, e.g., distance to origin
    candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        
        # Check if cand can be added to current_set
        valid = True
        for p in current_set:
            dx = cand[0] - p[0]
            dy = cand[1] - p[1]
            d2 = dx*dx + dy*dy
            if not _is_square(d2):
                valid = False
                break
        
        if not valid:
            continue
        
        # Check collinearity with any two points in current_set
        for i in range(len(current_set)):
            for j in range(i+1, len(current_set)):
                x1, y1 = current_set[i]
                x2, y2 = current_set[j]
                x3, y3 = cand
                # Determinant for collinearity
                det = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                if det == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Check concyclicity with any three points in current_set
        for i in range(len(current_set)):
            for j in range(i+1, len(current_set)):
                for k in range(j+1, len(current_set)):
                    x1, y1 = current_set[i]
                    x2, y2 = current_set[j]
                    x3, y3 = current_set[k]
                    x4, y4 = cand
                    
                    # Check if 4 points are concyclic
                    # Determinant of the matrix:
                    # x^2+y^2  x  y  1
                    # x1^2+y1^2 x1 y1 1
                    # x2^2+y2^2 x2 y2 1
                    # x3^2+y3^2 x3 y3 1
                    # x4^2+y4^2 x4 y4 1
                    # If det is 0, they are concyclic.
                    
                    def det4(m):
                        # Simple expansion for 4x4
                        return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[3][2]*m[2][3]) - m[1][2]*(m[2][1]*m[3][3]-m[3][1]*m[2][3]) + m[1][3]*(m[2][1]*m[3][2]-m[3][1]*m[2][2])) -
                                m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[3][2]*m[2][3]) - m[1][2]*(m[2][0]*m[3][3]-m[3][0]*m[2][3]) + m[1][3]*(m[2][0]*m[3][2]-m[3][0]*m[2][2])) +
                                m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[3][1]*m[2][3]) - m[1][1]*(m[2][0]*m[3][3]-m[3][0]*m[2][3]) + m[1][3]*(m[2][0]*m[3][1]-m[3][0]*m[2][1])) -
                                m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[3][1]*m[2][2]) - m[1][1]*(m[2][0]*m[3][2]-m[3][0]*m[2][2]) + m[1][2]*(m[2][0]*m[3][1]-m[3][0]*m[2][1])))
                    
                    M = [
                        [x1*x1+y1*y1, x1, y1, 1],
                        [x2*x2+y2*y2, x2, y2, 1],
                        [x3*x3+y3*y3, x3, y3, 1],
                        [x4*x4+y4*y4, x4, y4, 1]
                    ]
                    if det4(M) == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            current_set.append(cand)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_points = list(current_set)
            
            # If we have a large set, we might want to restart or continue
            # For now, we just continue
            # We can break if we reach 12, but let's see
            if len(current_set) == 12:
                break
    
    # Convert best_points to the required format
    formatted_points = []
    for p in best_points:
        x, y = p
        formatted_points.append([[x, 1], [y, 1]])
    
    return {'D': 1, 'points': formatted_points}
