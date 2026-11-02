"""Matrix multiplication 2x3x8 with rank-44 decomposition using block Strassen."""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    m1, m2 = 1, 2
    rank1 = n * m1 * p  # 16
    rank2 = 4 * 7  # 28
    rank = rank1 + rank2  # 44
    
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: m1=1, schoolbook
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = Fraction(1)
            V[0 * p + k][col] = Fraction(1)
            W[k * n + i][col] = Fraction(1)
            col += 1
    
    # Part 2: m2=2, Strassen for 2x2x8
    for kg in range(4):
        if time.time() > deadline - 5:
            break
        W00, W01, W10, W11 = 4*kg, 4*kg+1, 4*kg+2, 4*kg+3
        # U: a00=U[1], a01=U[2], a10=U[4], a11=U[5]
        # V: b00=V[8+2kg], b01=V[9+2kg], b10=V[16+2kg], b11=V[17+2kg]
        
        # m1 = (a00+a11)*(b00+b11)
        U[1][col] += Fraction(1); U[5][col] += Fraction(1)
        V[8+2*kg][col] += Fraction(1); V[17+2*kg][col] += Fraction(1)
        W[W00][col] += Fraction(1); W[W11][col] += Fraction(1)
        col += 1
        
        # m2 = (a10+a11)*b00
        U[4][col] += Fraction(1); U[5][col] += Fraction(1)
        V[8+2*kg][col] += Fraction(1)
        W[W10][col] += Fraction(1); W[W11][col] += Fraction(-1)
        col += 1
        
        # m3 = a00*(b01-b11)
        U[1][col] += Fraction(1)
        V[9+2*kg][col] += Fraction(1); V[17+2*kg][col] += Fraction(-1)
        W[W01][col] += Fraction(1); W[W11][col] += Fraction(1)
        col += 1
        
        # m4 = a11*(b10-b00)
        U[5][col] += Fraction(1)
        V[16+2*kg][col] += Fraction(1); V[8+2*kg][col] += Fraction(-1)
        W[W00][col] += Fraction(1); W[W10][col] += Fraction(1)
        col += 1
        
        # m5 = (a00+a01)*b11
        U[1][col] += Fraction(1); U[2][col] += Fraction(1)
        V[17+2*kg][col] += Fraction(1)
        W[W00][col] += Fraction(-1); W[W01][col] += Fraction(1)
        col += 1
        
        # m6 = (a10-a00)*(b00+b10)
        U[4][col] += Fraction(1); U[1][col] += Fraction(-1)
        V[8+2*kg][col] += Fraction(1); V[16+2*kg][col] += Fraction(1)
        W[W11][col] += Fraction(1)
        col += 1
        
        # m7 = (a01-a11)*(b01+b11)
        U[2][col] += Fraction(1); U[5][col] += Fraction(-1)
        V[9+2*kg][col] += Fraction(1); V[17+2*kg][col] += Fraction(1)
        W[W00][col] += Fraction(1); W[W01][col] += Fraction(1)
        col += 1
    
    assert col == rank
    
    def frac_to_num(x):
        if x.denominator == 1:
            return int(x)
        return float(x)
    
    return {
        "U": [[frac_to_num(x) for x in row] for row in U],
        "V": [[frac_to_num(x) for x in row] for row in V],
        "W": [[frac_to_num(x) for x in row] for row in W]
    }
