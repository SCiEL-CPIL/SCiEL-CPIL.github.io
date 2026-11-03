        def laderman_3x3x3():
            u = [[0]*11 for _ in range(9)]
            v = [[0]*11 for _ in range(9)]
            w = [[0]*11 for _ in range(9)]
            
            # 1. a00 b00
            u[0][0]=1; v[0][0]=1; w[0][0]=1
            # 2. a01 b10
            u[1][1]=1; v[3][1]=1; w[0][1]=1
            # 3. a02 b20
            u[2][2]=1; v[6][2]=1; w[0][2]=1
            # 4. a10 b01
            u[3][3]=1; v[1][3]=1; w[3][3]=1
            # 5. a11 b11
            u[4][4]=1; v[4][4]=1; w[3][4]=1
            # 6. a12 b21
            u[5][5]=1; v[7][5]=1; w[3][5]=1
            # 7. a20 b02
            u[6][6]=1; v[2][6]=1; w[6][6]=1
            # 8. a21 b12
            u[7][7]=1; v[5][7]=1; w[6][7]=1
            # 9. a22 b22
            u[8][8]=1; v[8][8]=1; w[6][8]=1
            
            # 10. (a01+a10) (b01+b10) ?
            # 11. (a02+a20) (b02+b20) ?
            
            # I will use Schoolbook for 3x3x3 (Rank 27) to be safe.
            # Rank = 4*11 + 4*27 = 152.
            # This is worse than 120.
            
            # I will use Strassen for 2x2x2 (Rank 7) and Schoolbook for the rest.
            # Rank 88 is the target.
            # I will implement Laderman correctly.
            
            return u, v, w
        