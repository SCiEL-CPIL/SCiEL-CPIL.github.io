def build_gq44():
    # GF(4) arithmetic
    # Elements: 0, 1, 2, 3 where 2=omega, 3=omega^2
    # omega^2 = omega + 1 => 3 = 2+1
    # omega^3 = 1 => 2*2=3, 2*3=1, 3*3=2
    
    def gf_add(a, b):
        return a ^ b  # XOR for GF(4) addition (since it's characteristic 2)
    
    def gf_mul(a, b):
        table = [
            [0,0,0,0],
            [0,1,2,3],
            [0,2,3,1],
            [0,3,1,2]
        ]
        return table[a][b]
    
    # Points: 85 points
    # Lines: 85 lines
    # ...
