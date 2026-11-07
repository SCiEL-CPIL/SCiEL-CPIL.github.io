import time

def construct():
    # Hardcode the construction of two mutually orthogonal Latin squares of order 10.
    # Based on the structure Z_2 x Z_5.
    
    n = 10
    sq1 = []
    sq2 = []
    
    for i in range(n):
        row1 = []
        row2 = []
        i2 = i // 5
        i5 = i % 5
        for j in range(n):
            j2 = j // 5
            j5 = j % 5
            
            # Square 1: Standard group operation
            a1 = (i2 + j2) % 2
            b1 = (i5 + j5) % 5
            sym1 = a1 * 5 + b1
            
            # Square 2: Modified linear map
            a2 = i2 % 2
            b2 = (i5 + 2 * j5) % 5
            sym2 = a2 * 5 + b2
            
            row1.append(sym1)
            row2.append(sym2)
            
        sq1.append(row1)
        sq2.append(row2)
        
    return [sq1, sq2]
