for ul, c in u_d.items():  # ul is 3x2x5 local: 2, 5
    i, jl = divmod(ul, 2)  # divmod(2,2)=(1,0), divmod(5,2)=(2,1)
    gu = i*4 + jl
