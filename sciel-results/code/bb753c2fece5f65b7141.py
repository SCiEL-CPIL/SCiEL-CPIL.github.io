import time

def construct():
    q = 29
    squares = {x * x % q for x in range(1, q)}
    
    n_base = q
    colors_base = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % q
            if diff in squares:
                colors_base.append(0)
            else:
                colors_base.append(1)
                
    return {'n': n_base, 'colors': colors_base}
