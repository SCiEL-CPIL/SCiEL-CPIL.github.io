import time

def construct():
    n = 10
    squares = []
    
    # Construct the first Latin Square
    # L[i][j] = (i + j) % n
    sq = []
    for i in range(n):
        row = []
        for j in range(n):
            val = (i + j) % n
            row.append(val)
        sq.append(row)
    
    squares.append(sq)
    
    # Final validation and return
    # Verify types
    final_squares = []
    for s in squares:
        clean_s = []
        for r in s:
            clean_r = []
            for v in r:
                if not isinstance(v, int) or isinstance(v, bool):
                    clean_r.append(int(v))
                else:
                    clean_r.append(v)
            clean_s.append(clean_r)
        final_squares.append(clean_s)
        
    return final_squares
