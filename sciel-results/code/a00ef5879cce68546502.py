import math

def construct():
    # Use zeros of J_{1/2}(x) = sqrt(2/(pi*x)) * sin(x)
    # J_{1/2}(x) = 0 when sin(x) = 0, i.e., x = pi*i for i = 1, 2, ...
    # For L_n^{(-1/2)}(x), the relevant Bessel order is -1/2 or 1/2
    # z_i = (pi * i)^2 for i = 1, ..., 12
    z = [(math.pi * i) ** 2 for i in range(1, 13)]
    return z
